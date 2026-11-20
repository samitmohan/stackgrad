import numpy as np

from . import functional as F
from . import tensor as T
from .module import Module
from .tensor import Tensor


class Linear(Module):
    """Fully-connected layer `y = x @ W + b` over the last dim, Kaiming init by default (`init='xavier'`)."""
    def __init__(self, in_features, out_features, bias=True, init='kaiming'):
        super().__init__()
        assert in_features > 0 and out_features > 0, "features must be positive"
        if init == 'xavier':
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        ) if bias else None

    def forward(self, x):
        assert x.shape[-1] == self.weight.shape[0], f"Linear expects last dim {self.weight.shape[0]}, got {x.shape[-1]}"
        # flatten to 2-d: one GEMM, and the weight grad needs no (batch, in, out) stack to sum
        rows = x.reshape(-1, x.shape[-1])
        if T._autocast is None:
            y = rows @ self.weight
        else:
            y = (rows.cast(T._autocast) @ self.weight.cast(T._autocast)).cast("float32")
        y = y.reshape(*x.shape[:-1], self.weight.shape[1])
        if self.bias is not None:
            y = y + self.bias
        return y


class ReLU(Module):
    """Rectified linear unit, `max(0, x)`."""
    def forward(self, x):
        return F.relu(x)


class Sigmoid(Module):
    """Logistic sigmoid, `1 / (1 + e**-x)`."""
    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Module):
    """Hyperbolic tangent."""
    def forward(self, x):
        return F.tanh(x)


class Softmax(Module):
    """Softmax over `axis`, turning logits into a probability distribution."""
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, axis=self.axis)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class LayerNorm(Module):
    """Normalize over the last dimension of any (..., num_features) input."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        assert num_features > 0, "num_features must be positive"
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        assert x.shape[-1] == self.gamma.shape[0], f"LayerNorm expects last dim {self.gamma.shape[0]}, got {x.shape[-1]}"
        mean = x.mean(axis=-1, keepdims=True)
        diff = x - mean
        var = (diff ** 2).mean(axis=-1, keepdims=True)
        x_hat = diff / (var + self.eps) ** 0.5
        return x_hat * self.gamma + self.beta


class GroupNorm(Module):
    """Normalize each group of channels of an (N, C, ...) input, then scale and shift each channel."""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0, "channels must split evenly into groups"
        self.num_groups, self.eps = num_groups, eps
        self.weight = Tensor(np.ones(num_channels, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(num_channels, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        g = x.reshape(x.shape[0], self.num_groups, -1)
        mean = g.mean(axis=-1, keepdims=True)
        diff = g - mean
        y = (diff / ((diff ** 2).mean(axis=-1, keepdims=True) + self.eps) ** 0.5).reshape(*x.shape)
        per_channel = (1, x.shape[1]) + (1,) * (len(x.shape) - 2)
        return y * self.weight.reshape(*per_channel) + self.bias.reshape(*per_channel)


class Embedding(Module):
    """Lookup table mapping integer ids to dense vectors."""
    def __init__(self, num_embeddings, embed_dim):
        super().__init__()
        assert num_embeddings > 0 and embed_dim > 0, "sizes must be positive"
        self.weight = Tensor(
            (np.random.randn(num_embeddings, embed_dim) * 0.02).astype(np.float32),
            requires_grad=True
        )

    def forward(self, idx):
        return self.weight[np.asarray(idx)]


class Dropout(Module):
    """Zero each activation with probability `p` during training, scaling the rest; a no-op in eval."""
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        return x * x._const(mask)


class BatchNorm1d(Module):
    """Normalize a `(batch, features)` input per feature, tracking running stats for eval."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        assert num_features > 0, "num_features must be positive"
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == self.gamma.shape[0], \
            f"Expected input shape (batch, {self.gamma.shape[0]}), got {x.shape}"

        if self._training:
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff ** 2).mean(axis=0, keepdims=True)
            x_hat = diff / (var + self.eps) ** 0.5

            # running stats stay host NumPy arrays on any device, so state_dict saves them as buffers
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.numpy().flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.numpy().flatten()
        else:
            rm = x._const(self.running_mean.reshape(1, -1))
            rv = x._const(self.running_var.reshape(1, -1))
            x_hat = (x - rm) / (rv + self.eps) ** 0.5

        return x_hat * self.gamma + self.beta


class RMSNorm(Module):
    """Divide each vector by its root-mean-square, then scale. LayerNorm without the mean and bias."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return x * ((x * x).mean(axis=-1, keepdims=True) + self.eps) ** -0.5 * self.weight


class DepthAttention(Module):
    """Attention Residuals: a layer's input is a learned softmax mix of earlier layers' outputs, not their sum."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.query = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.norm = RMSNorm(d_model, eps)

    def forward(self, sources):
        v = T.stack(sources)                                            # (N, B, T, D)
        scores = (self.norm(v) * self.query).sum(axis=-1, keepdims=True)  # (N, B, T, 1): keys normalized, values raw
        return (F.softmax(scores, axis=0) * v).sum(axis=0)


class SiLUMLP(Module):
    """Ungated feed-forward w2(silu(w1 x)), the baseline for the CS336 SwiGLU ablation."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class SwiGLU(Module):
    """Gated feed-forward from Llama and Qwen: w2(silu(w1 x) * w3 x), no biases."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)
        self.w3 = Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
