import unittest
from unittest.mock import patch
import torch
from experiments.position_encoding_audit.run import fixture
from experiments.position_encoding_audit.reference_mapping import anchored_coordinates, parse_references, VARIANTS


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class ReferenceMappingTest(unittest.TestCase):
    def test_fp32_cached_and_full_attention_with_reference_mapping(self):
        from qwenvl.models.configuration_qwen25_siglip import Qwen2Config
        from qwenvl.models.modeling_llm_eagar import Qwen25_SigLIPTextModel
        torch.manual_seed(21)
        config=Qwen2Config(vocab_size=50,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
            max_position_embeddings=128,cross_modal_position_type='girf',attention_dropout=0.,
            use_cache=True,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
        model=Qwen25_SigLIPTextModel(config).eval()
        _,pos,static,_,visual=fixture(3)
        ids=torch.arange(pos.shape[-1]).unsqueeze(0)%50
        valid=torch.ones_like(visual)
        ref=dict(start=pos.shape[-1]-2,end=pos.shape[-1]-1,entity=2,kind='question')
        parsed=dict(image_count=3,refs=[ref],question=ref)
        for variant in VARIANTS:
            def transform(p,v):
                return anchored_coordinates(p,v,variant,parsed,oracle_target=1)
            with torch.no_grad(),patch('qwenvl.models.modeling_llm_eagar.girf_token_coordinates',transform):
                full=model(input_ids=ids,position_ids=pos,static_position_ids=static,
                    visual_mask=visual,text_mask=~visual,valid_mask=valid,use_cache=False)
                pre=model(input_ids=ids[:,:-1],position_ids=pos[:,:,:-1],static_position_ids=static[:,:,:-1],
                    visual_mask=visual[:,:-1],text_mask=~visual[:,:-1],valid_mask=valid[:,:-1],use_cache=True)
                dec=model(input_ids=ids[:,-1:],position_ids=pos,static_position_ids=static,
                    visual_mask=visual,text_mask=~visual,valid_mask=valid,past_key_values=pre.past_key_values,
                    cache_position=torch.tensor([ids.shape[1]-1]),use_cache=True)
                torch.testing.assert_close(full.last_hidden_state[:,-1],dec.last_hidden_state[:,-1],atol=1e-6,rtol=1e-6)

    def test_parsed_mapping_ignores_external_target_label(self):
        _,p,_,_,v=fixture(3)
        ref=dict(start=p.shape[-1]-2,end=p.shape[-1]-1,entity=2,kind='question')
        d=dict(image_count=3,refs=[ref],question=ref)
        a=anchored_coordinates(p,v,'parsed_tail',d,oracle_target=0)
        b=anchored_coordinates(p,v,'parsed_tail',d,oracle_target=2)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_single_image_endpoint_all_conditions(self):
        _,p,_,_,v=fixture(1)
        ref=dict(start=p.shape[-1]-2,end=p.shape[-1]-1,entity=1,kind='question')
        d=dict(image_count=1,refs=[ref],question=ref)
        original=anchored_coordinates(p,v,'baseline',d)
        for variant in VARIANTS:
            torch.testing.assert_close(anchored_coordinates(p,v,variant,d,oracle_target=0),original,atol=0,rtol=0)

    def test_numeric_and_ordinal_parse_without_labels(self):
        for phrase in ("Image 2", "the second image"):
            s="Image 1: ## Image 2: ## What is in " + phrase + "?"
            d=parse_references(CharacterTokenizer(),torch.tensor(list(map(ord,s))),ord('#'))
            self.assertEqual(d['question']['entity'],2)
            self.assertEqual(d['image_count'],2)

    def test_rejects_ambiguous_and_out_of_range(self):
        for phrase in ("Image 1 and Image 2", "Image 3", "that picture"):
            s="Image 1: ## Image 2: ## What is in " + phrase + "?"
            with self.assertRaises(ValueError):
                parse_references(CharacterTokenizer(),torch.tensor(list(map(ord,s))),ord('#'))

    def test_only_text_entity_changes_and_cache_prefix_is_stable(self):
        _,p,_,_,v=fixture(3)
        d=dict(image_count=3,refs=[dict(start=0,end=1,entity=1,kind='header'),
            dict(start=p.shape[-1]-2,end=p.shape[-1]-1,entity=1,kind='question')])
        d['question']=d['refs'][-1]
        original=anchored_coordinates(p,v,'baseline',d)
        for variant in VARIANTS:
            c=anchored_coordinates(p,v,variant,d,oracle_target=0)
            torch.testing.assert_close(c[v],original[v],atol=0,rtol=0)
            torch.testing.assert_close(c[...,[0,2,3,4]],original[...,[0,2,3,4]],atol=0,rtol=0)
            pe=torch.cat([p,p[:,:,-1:]+1],dim=-1)
            ve=torch.cat([v,torch.zeros(1,1,dtype=torch.bool)],dim=-1)
            ce=anchored_coordinates(pe,ve,variant,d,oracle_target=0)
            torch.testing.assert_close(ce[:,:-1],c,atol=0,rtol=0)
        parsed=anchored_coordinates(p,v,'parsed_tail',d)
        oracle=anchored_coordinates(p,v,'oracle_tail',d,oracle_target=0)
        torch.testing.assert_close(parsed,oracle,atol=0,rtol=0)
        self.assertNotEqual(float(parsed[0,-1,1]),float(anchored_coordinates(p,v,'wrong_tail',d)[0,-1,1]))


if __name__ == '__main__':
    unittest.main()
