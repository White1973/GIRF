"""Experimental GIRF TV correction activated by weak adjacent entity separation."""
import inspect
import math
from contextlib import contextmanager
import torch
from experiments.girf.relative_entity_runtime import entity_indices
from experiments.girf.relative_origin_copy.runtime import _origins,_rotate


def modify(module,kwargs,anchors):
    meta=kwargs['relation_metadata'];result=_origins(module,meta,anchors)
    if result is None:return kwargs
    qrel,krel,_=result;qi,ki=entity_indices(meta)
    visual=meta['k_visual'];qvis=meta['q_visual'];kcoord=meta['k_girf'];qcoord=meta['q_girf']
    starts=visual.clone();pos=meta['k_pos']
    starts[1:] &= (~visual[:-1]) | (pos[1:]!=pos[:-1]+1)
    values=kcoord[starts,1].float()
    # In the factorial ablation the ordinal GIRF phase can be disabled while
    # this gate remains enabled.  Reconstruct the *same* fixed ordinal metric
    # from structural entity ids so gate activation is independent of whether
    # that metric is also present in the base rotary phase.
    if not bool(getattr(module.config,'girf_ordinal_enabled',True)):
        values=(2/math.pi)*torch.atan(ki[starts].float())
    delta=torch.empty_like(values);delta[0]=torch.inf
    if len(values)>1:delta[1:]=values[1:]-values[:-1]
    assert bool((delta[1:]>0).all())
    # Fixed conservative BF16-scale threshold, no fitted parameters.
    eps=torch.finfo(torch.bfloat16).eps
    weights=(1-delta/eps).clamp(0,1)
    if bool((weights==0).all()):return kwargs
    freq=module.rotary_emb.inv_freq.float();axis=torch.arange(freq.numel(),device=freq.device)%5
    qangle=qcoord[:,axis]*freq[None]+weights[qi,None]*qrel
    kangle=kcoord[:,axis]*freq[None]+weights[ki,None]*krel
    q=_rotate(kwargs['content_query'],qangle);k=_rotate(kwargs['content_key'],kangle)
    return dict(kwargs,static_query=torch.where(meta['q_text'][None,None,:,None],q,kwargs['static_query']),
                inter_key=torch.where(visual[None,None,:,None],k,kwargs['inter_key']))


@contextmanager
def installed():
    import qwenvl.models.modeling_llm_eagar as llm
    old_forward=llm.Qwen25_SigLIPTextModel.forward;old_attention=llm.decoupled_eager_attention_forward
    signature=inspect.signature(old_forward);context={};stats={'calls':0,'exact_original_calls':0}
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
        out=modify(module,kwargs,context['anchors']);stats['calls']+=1
        if out is kwargs:stats['exact_original_calls']+=1
        return old_attention(module,**out)
    llm.Qwen25_SigLIPTextModel.forward=forward;llm.decoupled_eager_attention_forward=attention
    try:yield stats
    finally:llm.Qwen25_SigLIPTextModel.forward=old_forward;llm.decoupled_eager_attention_forward=old_attention
