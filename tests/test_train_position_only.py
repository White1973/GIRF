import unittest
import torch
from experiments.position_encoding_audit.train_position_only import EntityPhase, training_cases
from qwenvl.models.girf_core import apply_girf_rotary


class TrainPositionOnlyTest(unittest.TestCase):
    def test_exact_initialization_and_gradients(self):
        torch.manual_seed(3)
        phase=EntityPhase();x=torch.randn(1,2,4,128);c=torch.randn(1,4,5);freq=torch.linspace(1,.01,64)
        original=apply_girf_rotary(x,c,freq);actual=phase(x,c,freq)
        torch.testing.assert_close(actual,original,atol=0,rtol=0)
        actual[...,0,:].sum().backward()
        self.assertGreater(float(phase.log_scale.grad.norm()),0)
        self.assertEqual(phase.log_scale.numel(),13)

    def test_spatial_channels_and_single_image_unchanged(self):
        phase=EntityPhase();phase.log_scale.data.fill_(1)
        x=torch.randn(1,2,4,128);c=torch.randn(1,4,5);freq=torch.linspace(1,.01,64)
        original=apply_girf_rotary(x,c,freq);actual=phase(x,c,freq)
        nonentity=(torch.arange(64)%5!=1).repeat(2)
        torch.testing.assert_close(actual[...,nonentity],original[...,nonentity],atol=0,rtol=0)
        c[...,1]=0
        torch.testing.assert_close(phase(x,c,freq),apply_girf_rotary(x,c,freq),atol=0,rtol=0)

    def test_budget_and_count_coverage(self):
        a=list(training_cases(48,'le4'));b=list(training_cases(48,'le8'))
        self.assertEqual(len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):
            self.assertEqual(len(x),2);self.assertEqual(len(y),2)
            self.assertEqual(x[0]['names'],x[1]['names'])
            self.assertNotEqual(x[0]['answer'],x[1]['answer'])
            if i%3!=2:self.assertEqual(x,y)
        self.assertEqual(max(len(c['names']) for pair in a for c in pair),4)
        self.assertEqual(max(len(c['names']) for pair in b for c in pair),8)


if __name__=='__main__':unittest.main()
