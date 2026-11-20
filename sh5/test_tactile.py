import unittest
import torch
from rho.policies.rho.tactile_encoders import SH5StateProjector
class TactileTests(unittest.TestCase):
    def test_gradients_shape_and_sensitivity(self):
        torch.manual_seed(905);m=SH5StateProjector(64)
        x=torch.randn(2,1,144,requires_grad=True);y=m(x)
        self.assertEqual(y.shape,(2,1,64));y.square().mean().backward()
        self.assertGreater(x.grad[...,54:].abs().sum().item(),0)
        z=x.detach().clone();z[...,54:]=0
        self.assertGreater((m(z)-y.detach()).abs().max().item(),1e-5)
        clone=SH5StateProjector(64);clone.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(clone(x.detach()),m(x.detach()))
    def test_reject_wrong_shape(self):
        with self.assertRaises(ValueError):SH5StateProjector(64)(torch.zeros(1,1,54))
if __name__=='__main__':unittest.main()
