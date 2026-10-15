import unittest
import torch
from tests.test_girf_relative_origin_copy import RelativeOriginTest
from experiments.girf.entity_resolution_gate.runtime import modify
from experiments.girf.relative_entity_runtime import entity_indices
from qwenvl.models.modeling_llm_eagar import repeat_kv

class ResolutionGateTest(unittest.TestCase):
 def test_exact_short_endpoint_and_long_direction(self):
  fixture=RelativeOriginTest()
  for n in [1,2,8]:
   module,kw,anchors=fixture.setup(blocks=n)
   self.assertIs(modify(module,kw,anchors),kw)
  module,kw,anchors=fixture.setup(blocks=11)
  result=modify(module,kw,anchors)
  self.assertIsNot(result,kw)
  visual=kw['relation_metadata']['k_visual'];old=kw['static_query']@repeat_kv(kw['inter_key'],2).transpose(-1,-2)
  new=result['static_query']@repeat_kv(result['inter_key'],2).transpose(-1,-2)
  vt=visual[:,None]&(~visual[None,:]);tv=(~visual[:,None])&visual[None,:]
  torch.testing.assert_close(new[...,vt],old[...,vt],atol=0,rtol=0)
  self.assertGreater((new[...,tv]-old[...,tv]).abs().max().item(),0)
  self.assertIs(result['dynamic_query'],kw['dynamic_query']);self.assertIs(result['key'],kw['key'])
 def test_query_uses_preceding_entity(self):
  fixture=RelativeOriginTest();module,kw,anchors=fixture.setup(blocks=11)
  qi,ki=entity_indices(kw['relation_metadata'])
  self.assertEqual(int(qi[5]),0) # text between first and second image
  self.assertEqual(int(qi[-1]),10)
  self.assertEqual(int(ki[6]),1)
 def test_gate_remains_active_without_ordinal_phase(self):
  fixture=RelativeOriginTest();module,kw,anchors=fixture.setup(blocks=11)
  module.config.girf_ordinal_enabled=False
  kw['relation_metadata']['q_girf'][:,1].zero_()
  kw['relation_metadata']['k_girf'][:,1].zero_()
  result=modify(module,kw,anchors)
  self.assertIsNot(result,kw)
  self.assertGreater((result['static_query']-kw['static_query']).abs().max().item(),0)
if __name__=='__main__':unittest.main()
