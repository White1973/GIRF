import unittest
from types import SimpleNamespace
import torch
from experiments.girf.position_factor_audit.runtime import factor_rotations,modify,rotate
from qwenvl.models.girf_core import apply_girf_rotary
from qwenvl.models.modeling_llm_eagar import apply_multimodal_rotary_pos_emb,repeat_kv

class FactorTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(3);self.f=1/(1e6**(torch.arange(64).float()/64))
  self.module=SimpleNamespace(rotary_emb=SimpleNamespace(inv_freq=self.f,attention_scaling=1.),rope_scaling={'mrope_section':[16,24,24]},config=SimpleNamespace(position_embedding_options={},position_embedding_type='mrope',sim2_rope_enabled=False))
  self.pos=torch.tensor([[[0,2,2,2,2,5]],[[0,2,2,3,3,5]],[[0,2,3,2,3,5]]]);self.anchor=torch.tensor([[[0,2,2,2,2,5]]]*3)
  self.visual=torch.tensor([False,True,True,True,True,False]);coords=torch.zeros(1,6,5);coords[0,1:5,1]=.5;coords[0,-1,1]=.5
  coords[0,1:5,3]=torch.tensor([-1,-1,1,1]);coords[0,1:5,4]=torch.tensor([-1,1,-1,1]);self.coords=coords
  self.q=torch.randn(1,4,6,128);self.k=torch.randn(1,2,6,128)
  def native(x,pos):
   angles=(pos.float()[:,:,:,None]*self.f);emb=torch.cat([angles,angles],-1)
   return apply_multimodal_rotary_pos_emb(x,emb.cos(),emb.sin(),[16,24,24])
  self.dq=native(self.q,self.anchor);self.dk=native(self.k,self.pos)
  self.kw=dict(content_query=self.q,content_key=self.k,static_query=apply_girf_rotary(self.q,coords,self.f),inter_key=apply_girf_rotary(self.k,coords,self.f),dynamic_query=native(self.q,self.pos),key=self.dk,relation_metadata=dict(q_pos=torch.arange(6),k_pos=torch.arange(6),q_girf=coords[0],k_girf=coords[0],q_text=~self.visual,k_visual=self.visual))
 def test_endpoints_and_all_direction_boundaries(self):
  old=self.kw['static_query']@repeat_kv(self.kw['inter_key'],2).transpose(-1,-2)
  v=self.visual;vt=v[:,None]&~v[None,:];tv=~v[:,None]&v[None,:]
  for n in range(8):
   b=f'{n:03b}';r=modify(self.module,self.kw,self.pos,self.anchor,b)
   score=r['static_query']@repeat_kv(r['inter_key'],2).transpose(-1,-2)
   torch.testing.assert_close(score[...,vt],old[...,vt],atol=0,rtol=0)
   self.assertIs(r['dynamic_query'],self.kw['dynamic_query']);self.assertIs(r['key'],self.kw['key'])
   if b=='000':
    torch.testing.assert_close(r['static_query'],self.kw['static_query'],atol=0,rtol=0);torch.testing.assert_close(r['inter_key'],self.kw['inter_key'],atol=0,rtol=0)
   if b=='111':
    expected=self.dq@repeat_kv(self.dk,2).transpose(-1,-2)
    torch.testing.assert_close(score[...,tv],expected[...,tv],atol=0,rtol=0)
 def test_origin_and_local_can_be_changed_independently(self):
  # Q bit must not alter K; K bits must not alter Q.
  for n in range(4):
   a=factor_rotations(self.module,self.kw,self.pos,self.anchor,'0'+f'{n:02b}')
   b=factor_rotations(self.module,self.kw,self.pos,self.anchor,'1'+f'{n:02b}')
   torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
  for qbit in ['0','1']:
   base=factor_rotations(self.module,self.kw,self.pos,self.anchor,qbit+'00')[0]
   for k in ['01','10','11']:torch.testing.assert_close(base,factor_rotations(self.module,self.kw,self.pos,self.anchor,qbit+k)[0],atol=0,rtol=0)
 def test_bfloat16_endpoints(self):
  self.q=self.q.bfloat16();self.k=self.k.bfloat16()
  kw=dict(self.kw,content_query=self.q,content_key=self.k)
  q,k=factor_rotations(self.module,kw,self.pos,self.anchor,'000')
  torch.testing.assert_close(q,apply_girf_rotary(self.q,self.coords,self.f),atol=0,rtol=0)
  torch.testing.assert_close(k,apply_girf_rotary(self.k,self.coords,self.f),atol=0,rtol=0)
if __name__=='__main__':unittest.main()
