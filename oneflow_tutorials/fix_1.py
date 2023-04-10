import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("../"))

import oneflow as torch
import oneflow.nn as nn
from flowvision.models import resnet18
import oneflow_pruning as tp


class MySlimmingPruner(tp.pruner.MetaPruner):
    def regularize(self, model, reg):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
                m.weight.grad.data.add_(reg*torch.sign(m.weight.data)) # Lasso for sparsity

class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        #note that we have multiple BNs in a group, 
        # we store layer-wise scores in a list and then reduce them to get the final results
        group_imp = [] # (num_bns, num_channels) 
        # 1. 遍历group以估计重要性
        for dep, idxs in group:
            layer = dep.target.module # 获取目标模型
            prune_fn = dep.handler    # 获取目标模型的剪枝函数，在本例中未使用
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data)
                group_imp.append(local_imp)
        if len(group_imp)==0: return None # 如果组中不包含BN层，则返回None
        # 2. 将组重要性减少到1-D分数向量。这里我们使用跨层的平均分数。
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0) 
        return group_imp # (num_channels, )

# 您可以实现任何重要性函数，只要它将组转换为1-D分数向量。

# 以下类是返回随机分数向量的重要性函数的示例。
class RandomImportance(tp.importance.Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs)) # 返回与idxs长度相同的随机分数向量。

model = resnet18(pretrained=True)
example_inputs = torch.randn(1, 3, 224, 224)


# 0. importance criterion 
imp = MySlimmingImportance()

# 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

# 2. Pruner initialization
iterative_steps = 5 # You can prune your model to the target sparsity iteratively.
pruner = MySlimmingPruner(
    model, 
    example_inputs, 
    global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=iterative_steps, # the number of iterations to achieve target sparsity
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)


base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()

    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # print(model)
    print(model(example_inputs).shape)
    print(
        "  Iter %d/%d, Params: %.2f M => %.2f M"
        % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
    )
    print(
        "  Iter %d/%d, MACs: %.2f G => %.2f G"
        % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
    )
    print("="*16)
    # finetune your model here
    # finetune(model)
    # ...