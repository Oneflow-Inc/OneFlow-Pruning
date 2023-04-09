import oneflow.nn as nn
from enum import IntEnum


class DummyMHA(nn.Module):
    def __init__(self):
        super(DummyMHA, self).__init__()


class _CustomizedOp(nn.Module):
    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return "CustomizedOp({})".format(str(self.op_cls))


class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_ConcatOp({})".format(self.offsets)


class _SplitOp(nn.Module):
    def __init__(self):
        super(_SplitOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_SplitOp({})".format(self.offsets)

class _ReshapeOp(nn.Module):
    def __init__(self):
        super(_ReshapeOp, self).__init__()

    def __repr__(self):
        return "_Reshape()"


class _ElementWiseOp(nn.Module):
    def __init__(self, grad_fn):
        super(_ElementWiseOp, self).__init__()
        self._grad_fn = grad_fn

    def __repr__(self):
        return "_ElementWiseOp({})".format(self._grad_fn)


######################################################
# Dummy Pruners
class DummyPruner(object):
    def __call__(self, layer, *args, **kargs):
        return layer

    def prune_out_channels(self, layer, idxs):
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return None

    def get_in_channels(self, layer):
        return None


class ConcatPruner(DummyPruner):
    pass

class ReshapePruner(DummyPruner):
    pass

class SplitPruner(DummyPruner):
    pass


class ElementWisePruner(DummyPruner):
    pass


# Define standard modules
TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_LAYERNORM = nn.modules.normalization.LayerNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear
TORCH_EMBED = nn.Embedding
TORCH_PARAMETER = nn.Parameter
TORCH_LSTM = nn.LSTM
try:
    TORCH_MHA = nn.MultiheadAttention
except:
    TORCH_MHA = DummyMHA  # for pytorch w/o MultiHeadAttention
TORCH_OTHERS = None

# Define operation types
class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    DEPTHWISE_CONV = 4
    CONCAT = 5  # torch.cat
    SPLIT = 6  # torch.split
    CUSTOMIZED = 7  # customized module
    ELEMENTWISE = 8  # element-wise add, sub, etc.
    LN = 9  # nn.LayerNorm
    EMBED = 10  # nn.Embedding
    PARAMETER = 11  # nn.Parameter
    MHA = 12
    LSTM = 13
    RESHAPE = 14

# Define module to operation type mapping
def module2type(module):
    mapping = {
        TORCH_CONV: OPTYPE.DEPTHWISE_CONV if module.groups == module.out_channels else OPTYPE.CONV,
        TORCH_BATCHNORM: OPTYPE.BN,
        TORCH_PRELU: OPTYPE.PRELU,
        TORCH_LINEAR: OPTYPE.LINEAR,
        _ConcatOp: OPTYPE.CONCAT,
        _SplitOp: OPTYPE.SPLIT,
        TORCH_LAYERNORM: OPTYPE.LN,
        TORCH_EMBED: OPTYPE.EMBED,
        _CustomizedOp: OPTYPE.CUSTOMIZED,
        nn.Parameter: OPTYPE.PARAMETER,
        TORCH_MHA: OPTYPE.MHA,
        TORCH_LSTM: OPTYPE.LSTM,
        _ReshapeOp: OPTYPE.RESHAPE,
    }
    return mapping.get(type(module), OPTYPE.ELEMENTWISE)

# Define operation type to class mapping
def type2class(op_type):
    mapping = {
        OPTYPE.CONV: TORCH_CONV,
        OPTYPE.DEPTHWISE_CONV: TORCH_CONV,
        OPTYPE.BN: TORCH_BATCHNORM,
        OPTYPE.PRELU: TORCH_PRELU,
        OPTYPE.LINEAR: TORCH_LINEAR,
        OPTYPE.CONCAT: _ConcatOp,
        OPTYPE.SPLIT: _SplitOp,
        OPTYPE.LN: TORCH_LAYERNORM,
        OPTYPE.EMBED: TORCH_EMBED,
        OPTYPE.CUSTOMIZED: _CustomizedOp,
        OPTYPE.PARAMETER: TORCH_PARAMETER,
        OPTYPE.MHA: TORCH_MHA,
        OPTYPE.LSTM: TORCH_LSTM,
        OPTYPE.RESHAPE: _ReshapeOp,
    }
    return mapping.get(op_type, _ElementWiseOp)
