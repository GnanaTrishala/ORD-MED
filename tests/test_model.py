import unittest
import torch
from config import Config
from models import build_model


class TestOrdMedNet(unittest.TestCase):
    """
    Unit tests to verify model architecture and forward pass sanity.
    """
    def setUp(self):
        # Configure standard config for testing
        self.config = Config()
        self.config.encoder.name = "efficientnet_b0"  # Small encoder for test speed
        self.config.encoder.pretrained = False       # No network download needed
        self.config.heads.use_evidential = True
        self.config.heads.dr_num_classes = 5
        self.config.heads.dme_num_classes = 3

    def test_forward_pass_shapes(self):
        """Tests if input propagates to correct output shapes."""
        # 1. Build model
        model = build_model(self.config)
        model.eval()

        # 2. Mock batch input (Batch size=2, Channels=3, Height=512, Width=512)
        dummy_input = torch.randn(2, 3, 512, 512)

        # 3. Propagate forward
        with torch.no_grad():
            outputs = model(dummy_input)

        # 4. Assert structure of outputs
        self.assertIn("dr_logits", outputs)
        self.assertIn("dme_logits", outputs)
        self.assertIn("shared_features", outputs)
        self.assertIn("dr_evidence", outputs)
        self.assertIn("dme_evidence", outputs)

        # 5. Assert shapes
        self.assertEqual(outputs["dr_logits"].shape, (2, 5))
        self.assertEqual(outputs["dme_logits"].shape, (2, 3))
        self.assertEqual(outputs["shared_features"].shape, (2, 512))
        self.assertEqual(outputs["dr_evidence"].shape, (2, 5))
        self.assertEqual(outputs["dme_evidence"].shape, (2, 3))
        
        # Verify evidential values are non-negative and >= 1 (Dirichlet alpha)
        self.assertTrue(torch.all(outputs["dr_evidence"] >= 1.0))
        self.assertTrue(torch.all(outputs["dme_evidence"] >= 1.0))


if __name__ == "__main__":
    unittest.main()
