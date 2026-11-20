import math

import numpy as np


class Function:
    differentiable = True

    def __init__(self, xp, *parents):
        self.xp = xp
        self.parents = parents
        self.needs_grad = [p.requires_grad for p in parents]

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def out_shape(self, *shapes, **kwargs):
        return np.broadcast_shapes(*shapes)

    def out_dtype(self, *dtypes, **kwargs):
        """Promote the input dtypes as the backend does, by adding 0-d probes."""
        out = dtypes[0]
        for d in dtypes[1:]:
            if d != out:
                out = (self.xp.asarray(0, dtype=out) + self.xp.asarray(0, dtype=d)).dtype
        return out


def _reduced(shape, axis, keepdims):
    axes = range(len(shape)) if axis is None else [a % len(shape) for a in (axis if isinstance(axis, (tuple, list)) else (axis,))]
    return tuple(1 if i in axes else d for i, d in enumerate(shape) if keepdims or i not in axes)


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Recip(Function):
    def forward(self, x):
        self.out = 1.0 / x
        return self.out

    def backward(self, grad):
        return (-grad * self.out * self.out,)


class Exp(Function):
    def forward(self, x):
        self.out = self.xp.exp(x)
        return self.out

    def backward(self, grad):
        return (grad * self.out,)


class Log(Function):
    def forward(self, x):
        self.x = x
        return self.xp.log(x)

    def backward(self, grad):
        return (grad / self.x,)


class Sigmoid(Function):
    # a primitive: composed from exp/recip, a very negative x gives 0 * inf = nan in backward
    def forward(self, x):
        self.out = self.xp.sigmoid(x)
        return self.out

    def backward(self, grad):
        return (grad * self.out * (1.0 - self.out),)


class Pow(Function):
    def forward(self, x, c):
        self.x, self.c = x, c
        return x ** c

    def backward(self, grad):
        return (grad * self.c * self.x ** (self.c - 1),)


class Cast(Function):
    def forward(self, x, dtype):
        self.dtype = x.dtype
        return x.astype(dtype)

    def backward(self, grad):
        return (grad.astype(self.dtype),)

    def out_dtype(self, x, dtype):
        return dtype


# binary ops: inputs that need a gradient arrive already expanded to the output shape

class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, grad):
        return grad, grad


class Mul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, grad):
        return (grad * self.y if self.needs_grad[0] else None,
                grad * self.x if self.needs_grad[1] else None)


class Where(Function):
    def forward(self, x, y, cond):
        self.cond = cond
        return self.xp.where(cond, x, y)

    def backward(self, grad):
        zero = self.xp.zeros_like(grad)
        return self.xp.where(self.cond, grad, zero), self.xp.where(self.cond, zero, grad)

    def out_dtype(self, x, y, cond):
        return super().out_dtype(x, y)


class Less(Function):
    differentiable = False

    def forward(self, x, y):
        return (x < y).astype(self.xp.float32)

    def out_dtype(self, x, y):
        return self.xp.float32


class Sum(Function):
    def forward(self, x, axis, keepdims):
        self.shape, self.axis, self.keepdims = x.shape, axis, keepdims
        return x.sum(axis=axis, keepdims=keepdims)

    def out_shape(self, x, axis, keepdims):
        return _reduced(x, axis, keepdims)

    def backward(self, grad):
        if self.axis is not None and not self.keepdims:
            grad = self.xp.expand_dims(grad, self.axis)
        return (self.xp.broadcast_to(grad, self.shape),)


class Max(Function):
    def forward(self, x, axis, keepdims):
        self.x, self.axis, self.keepdims = x, axis, keepdims
        self.out = x.max(axis=axis, keepdims=True)
        return self.out if keepdims else x.max(axis=axis)

    def out_shape(self, x, axis, keepdims):
        return _reduced(x, axis, keepdims)

    def backward(self, grad):
        # ties split the gradient evenly
        mask = (self.x == self.out).astype(self.x.dtype)
        mask = mask / mask.sum(axis=self.axis, keepdims=True)
        if self.axis is not None and not self.keepdims:
            grad = self.xp.expand_dims(grad, self.axis)
        return (mask * grad,)


class Reshape(Function):
    def forward(self, x, shape):
        self.shape = x.shape
        return x.reshape(shape)

    def backward(self, grad):
        return (grad.reshape(self.shape),)

    def out_shape(self, x, shape):
        size = math.prod(x)
        known = math.prod(d for d in shape if d != -1)
        out = tuple(size // known if d == -1 else d for d in shape)
        assert math.prod(out) == size, f"cannot reshape {x} into {shape}"
        return out


class Permute(Function):
    def forward(self, x, order):
        self.order = order
        return x.transpose(order)

    def backward(self, grad):
        inverse = [self.order.index(i) for i in range(len(self.order))]
        return (grad.transpose(inverse),)

    def out_shape(self, x, order):
        return tuple(x[i] for i in order)


class Expand(Function):
    """x already has the target rank; Tensor.expand adds the leading dims."""
    def forward(self, x, shape):
        self.shape = x.shape
        return self.xp.broadcast_to(x, shape)

    def backward(self, grad):
        return (unbroadcast(grad, self.shape),)

    def out_shape(self, x, shape):
        return tuple(shape)


class Index(Function):
    def forward(self, x, key):
        self.shape, self.key = x.shape, key
        return x[key]

    def backward(self, grad):
        # scatter_add: an index picked twice gets both gradients
        return (self.xp.scatter_add(self.shape, self.key, grad),)

    def out_shape(self, x, key):
        """Index a zero-stride stand-in of shape x; index arrays become zeros, bool masks stay (their count matters)."""
        def stand_in(k):
            if not hasattr(k, "dtype"):
                return k
            if "bool" in str(k.dtype):
                return self.xp.to_numpy(k)
            return np.zeros(tuple(k.shape), dtype=np.intp)
        key = tuple(map(stand_in, key)) if isinstance(key, tuple) else stand_in(key)
        # shortcut: a gather allocates one byte per result element here, compute the shape if it shows in a profile
        return np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), x, (0,) * len(x))[key].shape


class Cat(Function):
    def forward(self, *xs, axis):
        self.axis = axis
        self.bounds = []
        end = 0
        for x in xs:
            self.bounds.append((end, end + x.shape[axis]))
            end += x.shape[axis]
        return self.xp.concatenate(xs, axis=axis)

    def backward(self, grad):
        index = [slice(None)] * grad.ndim
        grads = []
        for start, end in self.bounds:
            index[self.axis] = slice(start, end)
            grads.append(grad[tuple(index)])
        return tuple(grads)

    def out_shape(self, *shapes, axis):
        axis %= len(shapes[0])
        return shapes[0][:axis] + (sum(s[axis] for s in shapes),) + shapes[0][axis + 1:]


class Im2col(Function):
    def forward(self, x, kh, kw, stride, padding):
        self.args = (x.shape, kh, kw, stride, padding)
        return self.xp.im2col(x, kh, kw, stride, padding)

    def backward(self, grad):
        return (self.xp.col2im(grad, *self.args),)

    def out_shape(self, x, kh, kw, stride, padding):
        n, c, h, w = x
        oh, ow = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
        return (n * oh * ow, c * kh * kw)


# a primitive: as broadcast-multiply-sum it would lose BLAS

class Matmul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        return x @ y

    def backward(self, grad):
        return (unbroadcast(grad @ self.y.swapaxes(-1, -2), self.x.shape) if self.needs_grad[0] else None,
                unbroadcast(self.x.swapaxes(-1, -2) @ grad, self.y.shape) if self.needs_grad[1] else None)

    def out_shape(self, x, y):
        """NumPy's matmul shape rule, 1-d operands included."""
        x1, y1 = len(x) == 1, len(y) == 1
        x, y = (1,) + x if x1 else x, y + (1,) if y1 else y
        assert x[-1] == y[-2], f"cannot multiply {x} by {y}"
        out = np.broadcast_shapes(x[:-2], y[:-2]) + (x[-2], y[-1])
        out = out[:-2] + out[-1:] if x1 else out
        return out[:-1] if y1 else out


class Attention(Function):
    """Causal GQA attention, a primitive so a backend can fuse it; backward rebuilds p from the saved lse."""
    def scores(self, q, k):
        xp = self.xp
        T, S = q.shape[2], k.shape[2]
        k = xp.repeat(k, q.shape[1] // k.shape[1], axis=1)
        s = q @ k.swapaxes(-1, -2) * q.shape[-1] ** -0.5
        if self.causal:  # query i sits at position S - T + i (kv cache)
            s = s + xp.triu(xp.full((T, S), -xp.inf, dtype=q.dtype), k=S - T + 1)
        if self.key_bias is not None:
            s = s + xp.asarray(self.key_bias, dtype=q.dtype)[:, None, None, :]
        return s

    def forward(self, q, k, v, key_bias=None, causal=True):
        """key_bias: optional (B, S) array added to each key's score (left-padding mask)."""
        self.key_bias, self.causal = key_bias, causal
        fused = getattr(self.xp, "fused_attention", None) if key_bias is None and causal else None  # the kernels take no mask
        out = fused(q, k, v) if fused else None
        self.p = None
        if out is None:
            s = self.scores(q, k)
            m = s.max(axis=-1, keepdims=True)
            p = self.xp.exp(s - m)
            l = p.sum(axis=-1, keepdims=True)
            self.p = p / l
            out = self.p @ self.xp.repeat(v, q.shape[1] // k.shape[1], axis=1), (m + self.xp.log(l))[..., 0]
        self.saved = q, k, v, out[0], out[1].astype(q.dtype)
        return out[0]

    def backward(self, grad):
        q, k, v, out, lse = self.saved
        xp = self.xp
        fused = getattr(xp, "fused_attention_backward", None)
        if fused is not None and self.p is None:  # p is None only after a fused forward
            return fused(q, k, v, out, lse, grad)
        B, Hkv, S, D = k.shape
        rep = q.shape[1] // Hkv
        p = self.p if self.p is not None else xp.exp(self.scores(q, k) - lse[..., None])
        dv = p.swapaxes(-1, -2) @ grad
        dp = grad @ xp.repeat(v, rep, axis=1).swapaxes(-1, -2)
        # softmax backward: ds = p * (dp - rowsum(p * dp)), and rowsum(p * dp) = rowsum(grad * out)
        ds = p * (dp - (grad * out).sum(axis=-1, keepdims=True)) * D ** -0.5
        dq = ds @ xp.repeat(k, rep, axis=1)
        dk = ds.swapaxes(-1, -2) @ q
        # each kv head sums the grads of the `rep` query heads it served
        return dq, dk.reshape(B, Hkv, rep, S, D).sum(axis=2), dv.reshape(B, Hkv, rep, S, D).sum(axis=2)

    def out_shape(self, q, k, v, key_bias=None, causal=True):
        return q[:-1] + v[-1:]

    def out_dtype(self, q, k, v, key_bias=None, causal=True):
        return q
