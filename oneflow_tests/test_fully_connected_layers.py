import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import unittest
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

class TestFullyConnectedNet(unittest.TestCase):
    def setUp(self):
        self.input_size = 128
        self.num_classes = 10
        self.hidden_units = 256
        self.model = FullyConnectedNet(self.input_size, self.num_classes, self.hidden_units)
        self.example_inputs = torch.randn(1, self.input_size)

    def test_pruning(self):
        # Build dependency graph
        DG = tp.DependencyGraph()
        DG.build_dependency(self.model, example_inputs=self.example_inputs)

        # get a pruning group according to the dependency graph.
        pruning_group = DG.get_pruning_group(self.model.fc1, tp.prune_linear_out_channels, idxs=[0, 4, 6])
        pruning_group.prune()
        
        output = self.model(self.example_inputs)
        self.assertEqual(output.shape, (1, 10))

class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""
    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc3 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        skip=x
        x = F.relu(self.fc2(x))
        x = x+skip 
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    unittest.main()

