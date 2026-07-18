import unittest
import torch
from config import Config
from models.losses.ordinal_loss import CORNLoss, EMDLoss, OrdinalLoss
from models.losses.evidential_loss import EvidentialLoss
from models.losses.multitask_loss import FocalLoss, DMELoss, MultiTaskLoss


class TestOrdinalLosses(unittest.TestCase):
    """Unit tests for Ordinal Loss modules (CORN and EMD)."""
    def setUp(self) -> None:
        self.num_classes = 5
        self.corn_loss = CORNLoss(self.num_classes)
        self.emd_loss = EMDLoss(self.num_classes)

    def test_corn_loss_shape_and_value(self) -> None:
        # B=2, K=5. Test both K and K-1 input logit shapes
        logits_k = torch.tensor([[2.0, 1.0, -1.0, -2.0, -3.0], 
                                 [-1.0, 2.0, 0.5, -0.5, -2.0]])
        logits_k_minus_1 = logits_k[:, :-1]
        targets = torch.tensor([2, 0])  # Class index 2 and Class index 0

        # Run CORN loss
        loss_k = self.corn_loss(logits_k, targets)
        loss_k_minus_1 = self.corn_loss(logits_k_minus_1, targets)

        # Assert output is scalar (0-dim tensor)
        self.assertEqual(loss_k.dim(), 0)
        self.assertEqual(loss_k_minus_1.dim(), 0)
        # Slicing the last logit should yield identical loss since it is discarded
        self.assertTrue(torch.allclose(loss_k, loss_k_minus_1))

    def test_emd_loss_shape(self) -> None:
        logits = torch.randn(4, self.num_classes)
        targets = torch.tensor([0, 4, 2, 1])

        loss = self.emd_loss(logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss >= 0.0)


class TestDmelLosses(unittest.TestCase):
    """Unit tests for DME specific loss components (BCE, CE, and Focal Loss)."""
    def test_focal_loss_binary_and_multiclass(self) -> None:
        # BCE-style (binary/multi-label) focal loss check
        logits_binary = torch.randn(3, 4)
        targets_binary = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0, 1.0],
                                       [1.0, 1.0, 0.0, 0.0]])
        focal_bce = FocalLoss(gamma=2.0)
        loss_bin = focal_bce(logits_binary, targets_binary)
        self.assertEqual(loss_bin.dim(), 0)

        # Multi-class focal loss check
        logits_mc = torch.randn(3, 4)
        targets_mc = torch.tensor([1, 3, 0])
        focal_mc = FocalLoss(gamma=2.0)
        loss_mc = focal_mc(logits_mc, targets_mc)
        self.assertEqual(loss_mc.dim(), 0)

    def test_dme_loss_configurations(self) -> None:
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        
        # Test BCE style
        dme_bce = DMELoss(num_classes=3, loss_type="bce")
        self.assertEqual(dme_bce(logits, targets).dim(), 0)

        # Test CE style
        dme_ce = DMELoss(num_classes=3, loss_type="ce")
        self.assertEqual(dme_ce(logits, targets).dim(), 0)

        # Test Focal style
        dme_focal = DMELoss(num_classes=3, loss_type="focal")
        self.assertEqual(dme_focal(logits, targets).dim(), 0)


class TestEvidentialLoss(unittest.TestCase):
    """Unit tests for Evidential Deep Learning Loss (Dirichlet subjective logic)."""
    def setUp(self) -> None:
        self.num_classes = 3
        self.edl_loss_mse = EvidentialLoss(self.num_classes, loss_type="mse")
        self.edl_loss_ce = EvidentialLoss(self.num_classes, loss_type="ce")

    def test_evidential_returns(self) -> None:
        # Evidential alpha parameters must be >= 1.0
        alpha = torch.rand(4, self.num_classes) + 1.0
        targets = torch.tensor([0, 2, 1, 0])

        # Test MSE Loss
        loss_mse, unc_mse = self.edl_loss_mse(alpha, targets, epoch=2, max_epochs=10)
        self.assertEqual(loss_mse.dim(), 0)
        self.assertEqual(unc_mse.shape, (4,))
        # Epistemic uncertainty u must be in range (0, 1]
        self.assertTrue(torch.all(unc_mse > 0.0))
        self.assertTrue(torch.all(unc_mse <= 1.0))

        # Test CE Loss
        loss_ce, unc_ce = self.edl_loss_ce(alpha, targets, epoch=2, max_epochs=10)
        self.assertEqual(loss_ce.dim(), 0)
        self.assertEqual(unc_ce.shape, (4,))


class TestMultiTaskLoss(unittest.TestCase):
    """Unit tests for the integrated MultiTaskLoss coordinator module."""
    def test_combined_multitask_loss(self) -> None:
        config = Config()
        config.heads.dr_num_classes = 5
        config.heads.dme_num_classes = 3
        config.heads.use_evidential = True
        
        criterion = MultiTaskLoss(config)

        # Setup mock outputs and targets
        outputs = {
            "dr_logits": torch.randn(4, 5),
            "dme_logits": torch.randn(4, 3),
            "dr_evidence": torch.rand(4, 5) + 1.0,
            "dme_evidence": torch.rand(4, 3) + 1.0
        }
        targets = {
            "dr_label": torch.tensor([0, 4, 2, 1]),
            "dme_label": torch.tensor([1, 0, 2, 1])
        }

        # Run forward pass
        loss_dict = criterion(outputs, targets, epoch=1)

        # Verify output keys
        expected_keys = [
            "loss", "dr_loss", "dme_loss", 
            "dr_evidential_loss", "dme_evidential_loss", 
            "evidential_loss", "dr_uncertainty", "dme_uncertainty"
        ]
        for key in expected_keys:
            self.assertIn(key, loss_dict)
            self.assertEqual(loss_dict[key].dim(), 0, f"Key '{key}' is not a scalar")


if __name__ == "__main__":
    unittest.main()
