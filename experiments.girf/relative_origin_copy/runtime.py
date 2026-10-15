"""Experimental TV encoding: previous-image to visual-origin displacement.

No target, answer, image-count threshold, or query-text absolute position is used.
Only batch-one, static-image, uncached full forward is supported.
"""
import inspect
from contextlib import contextmanager
import torch


def _rotate(x,angles):
    from qwenvl.models.girf_core import _rotate_half
    a=torch.cat([angles,angles],dim=-1)[None,None]
    return x*a.cos().to(x.dtype)+_rotate_half(x)*a.sin().to(x.dtype)


def _origins(module,meta,anchors):
    from qwenvl.models.modeling_llm_eagar import _axis_ids_for_pairs
    from experiments.girf.relative_entity_runtime import entity_indices
    assert torch.equal(meta['q_pos'],meta['k_pos'])
    visual=meta['k_visual'];pos=meta['k_pos']
    starts=visual.clone()
    starts[1:] &= (~visual[:-1]) | (pos[1:]!=pos[:-1]+1)
    ix=torch.where(starts)[0]
    if not len(ix):return None
    qi,ki=entity_indices(meta)
    assert int(ki[visual].max())==len(ix)-1
    origin=anchors[:,0,pos[ix]].T.float() # image × T/H/W
    pair_count=module.rotary_emb.inv_freq.numel();freq=module.rotary_emb.inv_freq.float()
    opts=dict(module.config.position_embedding_options or {})
    opts.setdefault('mrope_section',module.rope_scaling['mrope_section'])
    axes=_axis_ids_for_pairs('mrope',pair_count,opts,freq.device)
    phase=(origin-origin[0])[...,axes]*freq[None]
    return phase[qi],phase[ki],origin


def modify(module,kwargs,anchors,variant):
    assert variant in ['replace_e','add_e']
    meta=kwargs['relation_metadata']
    values=_origins(module,meta,anchors)
    if values is None:return kwargs
    qphase,kphase,origin=values
    # Exact mathematical identity when there is no inter-image displacement.
    # Keep original BF16 tensors as-is, avoiding recomputation roundoff.
    if bool((origin==origin[0]).all()):return kwargs
    freq=module.rotary_emb.inv_freq.float();count=freq.numel()
    axes=torch.arange(count,device=freq.device)%5
    qcoord=meta['q_girf'];kcoord=meta['k_girf']
    assert bool((qcoord[meta['q_text']][:,[0,2,3,4]]==0).all())
    assert bool((kcoord[meta['k_visual']][:,[0,2]]==0).all())
    k_local=torch.where(((axes==3)|(axes==4))[None],kcoord[:,axes]*freq[None],0.)
    qangles=qphase;kangles=kphase+k_local
    if variant=='add_e':
        qangles=qangles+torch.where((axes==1)[None],qcoord[:,axes]*freq[None],0.)
        kangles=kangles+torch.where((axes==1)[None],kcoord[:,axes]*freq[None],0.)
    q=_rotate(kwargs['content_query'],qangles)
    k=_rotate(kwargs['content_key'],kangles)
    return dict(kwargs,static_query=torch.where(meta['q_text'][None,None,:,None],q,kwargs['static_query']),
                inter_key=torch.where(meta['k_visual'][None,None,:,None],k,kwargs['inter_key']))


@contextmanager
def installed(variant):
    import qwenvl.models.modeling_llm_eagar as llm
    assert variant in ['replace_e','add_e']
    old_forward=llm.Qwen25_SigLIPTextModel.forward;old_attention=llm.decoupled_eager_attention_forward
    signature=inspect.signature(old_forward);context={};stats={'calls':0,'single_exact_calls':0}
    def forward(module,*args,**kwargs):
        bound=signature.bind(module,*args,**kwargs);bound.apply_defaults();m=bound.arguments
        assert not module.training and m['past_key_values'] is None and not m['use_cache']
        assert not context and m['position_ids'].shape[1]==1
        context['anchors']=m['static_position_ids']
        try:return old_forward(module,*args,**kwargs)
        finally:context.clear()
    def attention(module,*args,**kwargs):
        assert not args and module.cross_modal_position_type=='girf' and module.config.position_embedding_type=='mrope'
        assert not getattr(module.config,'sim2_rope_enabled',False)
        out=modify(module,kwargs,context['anchors'],variant)
        if out is kwargs:stats['single_exact_calls']+=1
        stats['calls']+=1
        return old_attention(module,**out)
    llm.Qwen25_SigLIPTextModel.forward=forward;llm.decoupled_eager_attention_forward=attention
    try:yield stats
    finally:llm.Qwen25_SigLIPTextModel.forward=old_forward;llm.decoupled_eager_attention_forward=old_attention
