import unittest
from types import SimpleNamespace
import torch
from experiments.girf.component_audit.directional import install
from qwenvl.models.girf_core import apply_girf_rotary


class DirectionalTest(unittest.TestCase):
    def test_only_named_cross_direction_changes(self):
        import qwenvl.models.modeling_llm_eagar as llm
        original=llm.decoupled_eager_attention_forward
        torch.manual_seed(8)
        q=torch.randn(1,2,4,128);k=torch.randn(1,1,4,128);f=torch.logspace(0,-4,64)
        visual=torch.tensor([False,True,True,False]);coords=torch.randn(1,4,5);coords[:,~visual,3:]=0
        qr=apply_girf_rotary(q,coords,f);kr=apply_girf_rotary(k,coords,f)
        base=qr@kr.repeat(1,2,1,1).transpose(-1,-2)
        module=SimpleNamespace(layer_idx=0,cross_modal_position_type='girf',rotary_emb=SimpleNamespace(inv_freq=f))
        kwargs=dict(static_query=qr,inter_key=kr,content_query=q,content_key=k,dynamic_query=q,key=k,
                    relation_metadata=dict(q_girf=coords[0],k_girf=coords[0],q_text=~visual,k_text=~visual))
        tv=(~visual[:,None])&visual[None,:];vt=visual[:,None]&(~visual[None,:])
        try:
            for direction,changed,kept in [('tv_only',tv,vt),('vt_only',vt,tv)]:
                llm.decoupled_eager_attention_forward=lambda module,**kw:kw
                install(dict(axis=4,band=0,scale=0.,start_layer=0,end_layer=12),direction)
                result=llm.decoupled_eager_attention_forward(module,**kwargs)
                scores=result['static_query']@result['inter_key'].repeat(1,2,1,1).transpose(-1,-2)
                torch.testing.assert_close(scores[...,kept],base[...,kept],rtol=0,atol=0)
                self.assertGreater((scores[...,changed]-base[...,changed]).abs().max().item(),0.)
                self.assertIs(result['dynamic_query'],q);self.assertIs(result['key'],k)
        finally:llm.decoupled_eager_attention_forward=original


if __name__=='__main__':unittest.main()
