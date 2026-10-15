"""Experimental cross-modal entity-phase correction; production files untouched."""
import inspect
import math
import torch


def entity_indices(meta):
    """Reconstruct legacy structural anchors from valid token boundaries."""
    if '_relative_entity_indices' in meta:
        return meta['_relative_entity_indices']
    visual, batch, pos = meta['k_visual'], meta['k_batch'], meta['k_pos']
    indices = torch.zeros_like(pos, dtype=torch.long)
    query = torch.zeros_like(meta['q_pos'], dtype=torch.long)
    for b in torch.unique(batch):
        mask = batch == b
        locations = torch.where(mask)[0]
        v, p = visual[mask], pos[mask]
        starts = v.clone()
        starts[1:] &= (~v[:-1]) | (p[1:] != p[:-1]+1)
        anchors = (starts.long().cumsum(0)-1).clamp_min(0)
        indices[locations] = anchors
        qm = meta['q_batch'] == b
        lookup = torch.searchsorted(p, meta['q_pos'][qm])
        if bool((lookup >= p.numel()).any()) or not torch.equal(p[lookup],meta['q_pos'][qm]):
            raise ValueError('Query positions must be present in full key metadata')
        query[qm] = anchors[lookup]
    meta['_relative_entity_indices'] = query, indices
    return query, indices


def correction(module, q, k, meta, start, end, scaling, scores, strength):
    if strength == 0:
        return scores
    if module.cross_modal_position_type != 'girf':
        raise ValueError('Relative entity candidate is only for legacy GIRF')
    qi, ki = entity_indices(meta)
    # Exact no-op for single/two-image inputs (including BF16 arithmetic).
    if int(ki.max()) <= 1:
        return scores
    half = q.shape[-1]//2
    pairs = torch.arange(1, half, 5, device=q.device)
    freq = module.rotary_emb.inv_freq[pairs].float()
    out = scores.clone()
    k0,k1 = k[...,pairs].float(),k[...,pairs+half].float()
    for entity in torch.unique(qi[start:end]):
        rows = torch.where(qi[start:end] == entity)[0]
        # All queries sharing this anchor have identical original E potential.
        original_q = meta['q_girf'][start+rows[0],1].float()
        old = meta['k_girf'][:,1].float()-original_q
        new = (2/math.pi)*torch.atan((ki-entity).float())
        delta = strength*(new-old)[:,None]*freq
        c,s = delta.cos()-1,delta.sin()
        # R(delta)K_old - K_old, on entity channels only; FP32 stable residual.
        dk0 = c*k0-s*k1
        dk1 = s*k0+c*k1
        qr=q[:,:,rows]
        change = (torch.matmul(qr[...,pairs].float(),dk0.transpose(-1,-2))
                  +torch.matmul(qr[...,pairs+half].float(),dk1.transpose(-1,-2)))*scaling
        cross = ((meta['q_visual'][start+rows,None] & meta['k_text'][None,:])
                 |(meta['q_text'][start+rows,None] & meta['k_visual'][None,:]))
        cross &= meta['q_batch'][start+rows,None] == meta['k_batch'][None,:]
        cross &= meta['q_pos'][start+rows,None] >= meta['k_pos'][None,:]
        out[:,:,rows] = (scores[:,:,rows].float()+torch.where(cross[None,None],change,0.)).to(scores.dtype)
    return out


def install(strength):
    """Insert one correction in an isolated copy of the actual attention code.

    Returns original function and exact copied source for provenance. Existing
    masking, chunk sizing, softmax, dropout and value aggregation stay intact.
    """
    if strength not in [0.,.25,1.]:
        raise ValueError('Only preregistered strengths are supported')
    import qwenvl.models.modeling_llm_eagar as llm
    original = llm.decoupled_eager_attention_forward
    source = inspect.getsource(original)
    marker = '        attn_dynamic_chunk = torch.matmul(q_dynamic_chunk, key.transpose(-1, -2).contiguous()) * scaling\n'
    assert source.count(marker)==1
    source = source.replace(marker, marker+'        attn_static_chunk = _entity_correction(module, q_static_chunk, inter_key, relation_metadata, i, end, scaling, attn_static_chunk)\n')
    namespace = dict(original.__globals__)
    namespace['_entity_correction'] = lambda *args: correction(*args,strength=strength)
    exec(compile(source,'<isolated_relative_entity_attention>','exec'),namespace)
    llm.decoupled_eager_attention_forward = namespace['decoupled_eager_attention_forward']
    return original,source
