import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("../"))

import torch
from torchvision.models import resnet18
import torch_pruning as tp

# 0. prepare your model and example inputs
model = resnet18(pretrained=True).eval()
example_inputs = torch.randn(1,3,224,224)

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# 2. Select some channels to prune. Here we prune the channels indexed by [2, 6, 9].
pruning_idxs = pruning_idxs=[2, 6, 9]
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

print(group[0])

print("Dep:", group[1][0])
print("Indices:", group[1][1])

print("Source Node:", group[1][0].source.module) # group[1][0].source.module # get the nn.Module
print("Target Node:", group[1][0].target.module) # group[1][0].target.module # get the nn.Module
print("Trigger Function:", group[1][0].trigger)
print("Handler Function:", group[1][0].handler)


idx = group[0][1]
dep = group[0][0]
print(f'{idx=}')
dep(idx)

model = resnet18(pretrained=True).eval()
example_inputs = torch.randn(1,3,224,224)

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# 2. Select some channels to prune. Here we prune the channels indexed by [2, 6, 9].
pruning_idxs = pruning_idxs=[2, 6, 9]
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

group.prune()

print(model(torch.randn(1,3,224,224)).shape)