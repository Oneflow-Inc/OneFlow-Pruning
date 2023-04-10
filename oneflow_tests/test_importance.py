import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import unittest
import oneflow as torch
from flowvision.models import resnet18
import oneflow_pruning as tp

class TestImportance(unittest.TestCase):
    def setUp(self):
        self.model = resnet18(pretrained=True)
        self.DG = tp.DependencyGraph()
        self.example_inputs = torch.randn(1,3,224,224)
        self.DG.build_dependency(self.model, example_inputs=self.example_inputs)
        self.pruning_idxs = list( range( self.DG.get_out_channels(self.model.conv1) ))
        self.pruning_group = self.DG.get_pruning_group( self.model.conv1, tp.prune_conv_out_channels, idxs=self.pruning_idxs)

    def test_random_importance(self):
        random_importance = tp.importance.RandomImportance()
        rand_imp = random_importance(self.pruning_group)
        print("Random: ", rand_imp)

    def test_magnitude_importance_l1(self):
        magnitude_importance = tp.importance.MagnitudeImportance(p=1)
        mag_imp = magnitude_importance(self.pruning_group)
        print("L-1 Norm, Group Mean: ", mag_imp)

    def test_magnitude_importance_l2_mean(self):
        magnitude_importance = tp.importance.MagnitudeImportance(p=2)
        mag_imp = magnitude_importance(self.pruning_group)
        print("L-2 Norm, Group Mean: ", mag_imp)

    def test_magnitude_importance_l2_sum(self):
        magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction='sum')
        mag_imp = magnitude_importance(self.pruning_group)
        print("L-2 Norm, Group Sum: ", mag_imp)

    def test_magnitude_importance_l2_no_reduction(self):
        magnitude_importance = tp.importance.MagnitudeImportance(p=2, group_reduction=None)
        mag_imp = magnitude_importance(self.pruning_group)
        print("L-2 Norm, No Reduction: ", mag_imp)

    def test_bn_scale_importance(self):
        bn_scale_importance = tp.importance.BNScaleImportance()
        bn_imp = bn_scale_importance(self.pruning_group)
        print("BN Scaling, Group mean: ", bn_imp)

    def test_lamp_importance(self):
        lamp_importance = tp.importance.LAMPImportance()
        lamp_imp = lamp_importance(self.pruning_group)
        print("LAMP: ", lamp_imp)

if __name__ == '__main__':
    unittest.main()


