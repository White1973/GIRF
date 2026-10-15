import unittest
from types import SimpleNamespace
import torch
from experiments.girf.relative_entity_runtime import entity_indices,correction,install
from experiments.girf.relative_entity_design import pairwise_entity_score
from qwenvl.models.girf_core import girf_token_coordinates,apply_girf_rotary


def fixture():
    visual=torch.tensor([0,1,1,0,1,1,0,1,1,0],dtype=torch.bool)
    n=visual.numel();pos=torch.arange(n);coords=girf_token_coordinates(pos[None,None].expand(3,1,n),visual[None])[0]
    meta=dict(k_visual=visual,k_text=~visual,q_visual=visual,q_text=~visual,
              k_pos=pos,q_pos=pos,k_batch=torch.zeros(n,dtype=torch.long),q_batch=torch.zeros(n,dtype=torch.long),
              k_girf=coords,q_girf=coords)
    return meta


class RuntimeTest(unittest.TestCase):
    def test_structural_anchors_and_cached_query(self):
        m=fixture();q,k=entity_indices(m)
        self.assertEqual(k.tolist(),[0,0,0,0,1,1,1,2,2,2])
        self.assertTrue(torch.equal(q,k))
        m=fixture()
        for name in list(m):
            if name.startswith('q_'):m[name]=m[name][-1:]
        self.assertEqual(entity_indices(m)[0].tolist(),[2])

    def test_batch_resets_anchors(self):
        m=fixture()
        for name in list(m):m[name]=torch.cat([m[name],m[name]])
        m['k_batch'][10:]=1;m['q_batch'][10:]=1
        q,k=entity_indices(m)
        self.assertEqual(k[:10].tolist(),k[10:].tolist())

    def test_correction_matches_pairwise_reference_and_preserves_other_edges(self):
        torch.manual_seed(8);m=fixture();freq=torch.logspace(0,-4,64)
        rawq=torch.randn(1,2,10,128);rawk=torch.randn(1,2,10,128)
        q=apply_girf_rotary(rawq,m['q_girf'][None],freq);k=apply_girf_rotary(rawk,m['k_girf'][None],freq)
        module=SimpleNamespace(cross_modal_position_type='girf',rotary_emb=SimpleNamespace(inv_freq=freq))
        scores=q@k.transpose(-1,-2);qi,ki=entity_indices(m);ix=torch.arange(1,64,5)
        qpair=torch.stack([rawq[0,:,:,ix],rawq[0,:,:,ix+64]],dim=-1)
        kpair=torch.stack([rawk[0,:,:,ix],rawk[0,:,:,ix+64]],dim=-1)
        cross=(m['q_visual'][:,None]!=m['k_visual'][None,:]) & (m['q_pos'][:,None]>=m['k_pos'][None,:])
        for strength in [0.,.25,1.]:
            result=correction(module,q,k,m,0,10,1.,scores,strength)
            delta=pairwise_entity_score(qpair,kpair,qi,ki,freq[ix],strength)-pairwise_entity_score(qpair,kpair,qi,ki,freq[ix],0.)
            expected=scores+torch.where(cross[None,None],delta[None],0.)
            torch.testing.assert_close(result,expected,rtol=2e-5,atol=1e-5)
            torch.testing.assert_close(result[...,~cross],scores[...,~cross],rtol=0,atol=0)

    def test_copied_attention_and_chunked_cache(self):
        import qwenvl.models.modeling_llm_eagar as llm
        torch.manual_seed(9);m=fixture();freq=torch.logspace(0,-4,64)
        rawq=torch.randn(1,2,10,128);rawk=torch.randn(1,2,10,128);value=torch.randn_like(rawk)
        q=apply_girf_rotary(rawq,m['q_girf'][None],freq);k=apply_girf_rotary(rawk,m['k_girf'][None],freq)
        module=SimpleNamespace(cross_modal_position_type='girf',rotary_emb=SimpleNamespace(inv_freq=freq),
                               num_heads=2,num_key_value_groups=1,training=False,config=SimpleNamespace(attention_chunk_elements=200))
        args=dict(module=module,static_query=q,dynamic_query=rawq,key=rawk,value=value,
                  inter_attention_mask=None,intra_attention_mask=None,inter_key=k,scaling=128**-.5,relation_metadata=m)
        orig,source=install(0.)
        try:
            expected=orig(**args)[0];actual=llm.decoupled_eager_attention_forward(**args)[0]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        finally:llm.decoupled_eager_attention_forward=orig
        orig,_=install(1.)
        try:
            full=llm.decoupled_eager_attention_forward(**args)[0]
            cached={key:(v[-1:] if key.startswith('q_') else v) for key,v in fixture().items()}
            args.update(static_query=q[:,:,-1:],dynamic_query=rawq[:,:,-1:],relation_metadata=cached)
            tail=llm.decoupled_eager_attention_forward(**args)[0]
            torch.testing.assert_close(full[:,-1:],tail,rtol=1e-5,atol=1e-6)
        finally:llm.decoupled_eager_attention_forward=orig


if __name__=='__main__':unittest.main()
