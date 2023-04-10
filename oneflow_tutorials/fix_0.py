import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("/home/fengwen/OneFlow-Pruning"))

import oneflow as torch
from flowvision.models import resnet18
import oneflow_pruning as tp

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

print(pruning_group)

all_groups = list(DG.get_all_groups())
print("Number of Groups: %d"%len(all_groups))
print("The last Group:", all_groups[-1])


model = resnet18(pretrained=True)
example_inputs = torch.randn(1, 3, 224, 224)

# 0. importance criterion for parameter selections
# 定义参数选择的重要性标准
imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

# 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

        
# 2. Pruner initialization
# 剪枝器初始化
iterative_steps = 5 # You can prune your model to the target sparsity iteratively.
pruner = tp.pruner.MagnitudePruner(
    model, 
    example_inputs, 
    global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=iterative_steps, # the number of iterations to achieve target sparsity
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)
# 计算模型的基本计算量和参数数量
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    # 3. the pruner.step will remove some channels from the model with least importance
    # pruner.step 将从具有最小重要性的模型中删除一些通道
    pruner.step()
    
    # 4. Do whatever you like here, such as fintuning
    # 在此处进行微调等操作
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print(model(example_inputs).shape)
    print(
        "  Iter %d/%d, Params: %.2f M => %.2f M"
        % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
    )
    print(
        "  Iter %d/%d, MACs: %.2f G => %.2f G"
        % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
    )
    # finetune your model here
    # finetune(model)
    # ...

