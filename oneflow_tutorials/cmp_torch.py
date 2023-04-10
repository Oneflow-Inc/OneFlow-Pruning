import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("/home/fengwen/OneFlow-Pruning"))

import torch
from torchvision.models import resnet18
import torch_pruning as tp


# 0. 准备模型和示例输入
model = resnet18(pretrained=True).eval()
example_inputs = torch.randn(1,3,224,224)

# 1. 为resnet18构建依赖图
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# 2. 选择要修剪的一些通道。这里我们修剪索引为[2, 6, 9]的通道。
pruning_idxs = pruning_idxs=[2, 6, 9]
pruning_group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs )

# 3. 修剪与model.conv1耦合的所有分组层
if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

print("After pruning:")
print(model)