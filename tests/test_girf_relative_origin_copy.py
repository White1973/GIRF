import unittest
from types import SimpleNamespace
import torch
from experiments.girf.relative_origin_copy.runtime import modify,_origins
from qwenvl.models.girf_core import apply_girf_rotary
from qwenvl.models.modeling_llm_eagar import repeat_kv

class RelativeOriginTest(unittest.TestCase):
 def setup(self,blocks=2,prefix=0):
  torch.manual_seed(5);n=blocks*4+blocks+2
  v=[];anchors=[];coords=[]
  v.append(False);anchors.append(prefix);coords.append([0,0,0,0,0])
  for i in range(blocks):
   start=prefix+i*10+1
   v.extend([True]*4);anchors.extend([start]*4)
   e=2/torch.pi*torch.atan(torch.tensor(float(i)))
   coords.extend([[0,float(e),0,y,x] for y,x in [(-1,-1),(-1,1),(1,-1),(1,1)]])
   v.append(False);anchors.append(start+4);coords.append([0,float(e),0,0,0])
  v.append(False);anchors.append(prefix+blocks*10+9);coords.append([0,float(e),0,0,0])
  assert len(v)==n
  visual=torch.tensor(v);coord=torch.tensor(coords).float()[None]
  f=1/(1e6**(torch.arange(64).float()/64));q=torch.randn(1,4,n,128);k=torch.randn(1,2,n,128)
  module=SimpleNamespace(rotary_emb=SimpleNamespace(inv_freq=f),rope_scaling={'mrope_section':[16,24,24]},config=SimpleNamespace(position_embedding_options={}))
  pos=torch.arange(n);static=torch.stack([torch.tensor(anchors)]*3)[:,None]
  meta=dict(q_pos=pos,k_pos=pos,q_batch=torch.zeros(n,dtype=torch.long),k_batch=torch.zeros(n,dtype=torch.long),q_text=~visual,k_text=~visual,q_visual=visual,k_visual=visual,q_girf=coord[0],k_girf=coord[0])
  kw=dict(content_query=q,content_key=k,static_query=apply_girf_rotary(q,coord,f),inter_key=apply_girf_rotary(k,coord,f),dynamic_query=q,key=k,relation_metadata=meta)
  return module,kw,static
 def test_single_exact_identity_both_variants(self):
  m,kw,origin=self.setup(blocks=1)
  for v in ['replace_e','add_e']:self.assertIs(modify(m,kw,origin,v),kw)
 def test_multi_direction_and_previous_anchor(self):
  m,kw,origin=self.setup(blocks=2)
  meta=kw['relation_metadata'];v=meta['q_visual']
  old=kw['static_query']@repeat_kv(kw['inter_key'],2).transpose(-1,-2)
  tv=(~v[:,None])&v[None,:];vt=v[:,None]&(~v[None,:])
  for version in ['replace_e','add_e']:
   out=modify(m,kw,origin,version);new=out['static_query']@repeat_kv(out['inter_key'],2).transpose(-1,-2)
   torch.testing.assert_close(new[...,vt],old[...,vt],atol=0,rtol=0)
   self.assertGreater((new[...,tv]-old[...,tv]).abs().max().item(),0)
   self.assertIs(out['dynamic_query'],kw['dynamic_query']);self.assertIs(out['key'],kw['key'])
  qphase,kphase,images=_origins(m,meta,origin)
  self.assertTrue(bool((qphase[5]==0).all())) # text between first and second images
  torch.testing.assert_close(qphase[-1],kphase[6],atol=0,rtol=0) # last query anchors second visual block
  shifted=origin+2048
  q2,k2,_=_origins(m,meta,shifted)
  torch.testing.assert_close(qphase,q2,atol=0,rtol=0);torch.testing.assert_close(kphase,k2,atol=0,rtol=0)
if __name__=='__main__':unittest.main()
