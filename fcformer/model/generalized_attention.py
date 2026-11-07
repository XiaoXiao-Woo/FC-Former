"""Generalized self-attention (Section III-A of the paper).

Equation and theorem numbers refer to:
    X. Wu, Z.-H. Cao, T.-Z. Huang, L.-J. Deng, J. Chanussot, G. Vivone,
    "Fully-Connected Transformer for Multi-Source Image Fusion", IEEE TPAMI 47(3), 2025.

Theorem 2 is one operation on factors Q, K, V, and a reordering vector i with a mode k selects the
form of attention. `GeneralizedAttention` holds that operation and takes i and k when it is called.
With d heads, P blocks and the spatial dimensions merged into S, the factors are B x d x P x S x C,
the order of Section III-A, or B x d x P x L when S and C are merged into L = S * C. The pair (i, k)
selects the form:

    Spa-SA    i = (1, 2, 3, 4, 5), k = 5   BdP x S x C   A in R^{BdP x S_Q x S_K}
    CAL-SA    i = (1, 2, 3, 5, 4), k = 5   BdP x C x S   A in R^{BdP x C x C}       P = 1
    Patch-SA  i = (1, 2, 3, 4),    k = 4   Bd x P x SC   A in R^{Bd x P x P}

Reordering vectors and modes are 1-based, as in Definition 2. Unfolding along mode k gives
J1 x J2 x J3 with J3 = I[i_k], and with J2 = I[i_(N-1)] when k = N.

The examples of Section III-A write each attention on its own layout of the factors, so their vectors
differ from the ones above: channel self-attention on B x d x C x S with i = (1, 2, 3, 4), k = 4 is
CAL-SA above with P = 1, and patch self-attention on B x d x P x S x C unfolded to Bd x P x SC is
Patch-SA above. One layout for all forms lets them be summed in one layer (Algorithm 1).

Eq. (5) is the general form and leaves some details open. FC-Former makes these choices, each
explained where it is coded:
    - the score is a cosine similarity with a learned temperature per head (GeneralizedAttention);
    - the projections W_Q, W_K, W_V act on the channels (InScaleAttention, CrossScaleAttention);
    - cross-scale attention also has a local convolution path (CrossScaleAttention.local_path);
    - the residual and the feed-forward step form the layer around the attention (fcformer.py, TransformerLayer).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_based(reordering, order):
    if sorted(reordering) != list(range(1, order + 1)):
        raise ValueError(f"{tuple(reordering)} is not a reordering of 1..{order}")
    return [index - 1 for index in reordering]


def _unfolding_layout(n, k, order):
    """Dimension order before the reshape, and how many leading dimensions form J1."""
    if not 1 < k <= order:
        raise ValueError(f"mode k must satisfy 1 < k <= {order}, got {k}")
    n = _zero_based(n, order)
    if k < order:
        return n[:k - 1] + n[k:] + [n[k - 1]], k - 1
    return n, order - 2


def batched_unfold(x, n, k):
    """Definition 2, Eq. (3): the batched mode-k unfolding X_[n;k], of size J1 x J2 x J3."""
    index, num_batch = _unfolding_layout(n, k, x.dim())
    x = x.permute(index)
    sizes = x.shape
    return x.reshape(math.prod(sizes[:num_batch]), math.prod(sizes[num_batch:-1]), sizes[-1])


def batched_fold(matrix, n, k, shape):
    """Inverse of `batched_unfold`: a tensor of the given shape, in its original index order."""
    index, _ = _unfolding_layout(n, k, len(shape))
    x = matrix.reshape([shape[i] for i in index])
    inverse = [0] * len(index)
    for position, i in enumerate(index):
        inverse[i] = position
    return x.permute(inverse)


def batched_tensor_product(x, y, m, n, k):
    """Definition 3, Eq. (4), and Theorem 1: X x^m_n Y = X_[m;k] Y_[n;k]^T."""
    return batched_unfold(x, m, k) @ batched_unfold(y, n, k).transpose(1, 2)


def tensor_blocking(x, q):
    """Definition 1, Eq. (2), with stride s = q and padding p = 0.

    [B, C, H, W] -> [B, P, q, q, C], with P = (H // q) * (W // q).
    The reshaping is that of window_partition in Swin Transformer (see THIRD_PARTY_NOTICES.md).
    """
    b, c, h, w = x.shape
    if h % q or w % q:
        raise ValueError(f"a feature map of {h} x {w} cannot be cut into blocks of {q} x {q}: choose another image size")
    x = x.view(b, c, h // q, q, w // q, q).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, -1, q, q, c)


def tensor_unblocking(t, h, w):
    """Inverse of `tensor_blocking`: [B, P, q, q, C] -> [B, C, H, W]."""
    b, _, q, _, c = t.shape
    x = t.view(b, h // q, w // q, q, q, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class GeneralizedAttention(nn.Module):
    """Theorem 2, Eq. (5), on factors Q, K, V: returns A x V folded back to the shape of Q.

    The module holds only the part that the reordering vector and the mode decide. The caller builds
    Q, K, V and the residual, which sets the projections and the layer form, and passes i and k:

        attention(q, key, value, i=(1, 2, 3, 4, 5), k=5)     # Spa-SA
        attention(q, key, value, i=(1, 2, 3, 5, 4), k=5)     # CAL-SA
        attention(q, key, value, i=(1, 2, 3, 4), k=4)        # Patch-SA

    Score of FC-Former ("cosine"): Q and K are l2-normalized along J3 and their product is multiplied
    by a learned temperature per head, A = Softmax(t_d * Q_n K_n^T). A cosine lies in [-1, 1] whatever
    the magnitude of the features, and t_d sets how sharp each head's Softmax is. Eq. (5) writes the
    general scaling QK^T / sqrt(d), with d the length of the vectors of Q and K (d = J3 here),
    available as "scaled_dot". The published weights use "cosine".

    Args:
        num_heads: d, for the learned temperature of the "cosine" score.
        score: "cosine" or "scaled_dot", see above.
        attn_drop: dropout on A.
    """

    def __init__(self, num_heads=None, score="cosine", attn_drop=0.0):
        super().__init__()
        self.score = score
        if score == "cosine":
            if num_heads is None:
                raise ValueError('the "cosine" score needs num_heads, for one temperature per head')
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        elif score != "scaled_dot":
            raise ValueError(f"unknown score {score}")
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop != 0.0 else nn.Identity()

    def _apply_temperature(self, scores, q, i, k, head_mode):
        """Multiply each head by its temperature: J1 is opened so that the heads can broadcast."""
        index, num_batch = _unfolding_layout(i, k, q.dim())
        batch_modes = index[:num_batch]
        if head_mode - 1 not in batch_modes:
            raise ValueError(f"head mode {head_mode} is not in J1 for i={tuple(i)}, k={k}")
        sizes = [q.size(m) for m in batch_modes]
        scores = scores.view(*sizes, *scores.shape[1:])
        trailing = len(sizes) - 1 - batch_modes.index(head_mode - 1) + 2
        return (scores * self.temperature.view(-1, *[1] * trailing)).flatten(0, len(sizes) - 1)

    def forward(self, q, key, value, i, k, head_mode=2):
        """
        Args:
            q, key, value: the factors Q, K, V.
            i: reordering vector of the unfolding.
            k: the mode, 1 < k <= N.
            head_mode: which mode of the factors holds the heads (mode 2 in B x d x P x ...).
                It must fall inside J1.
        """
        q_i = batched_unfold(q, i, k)
        key_i = batched_unfold(key, i, k)
        value_i = batched_unfold(value, i, k)

        if q_i.size(-1) != key_i.size(-1):
            raise ValueError(
                f"Q and K differ in J3 ({q_i.size(-1)}, {key_i.size(-1)}) for i={tuple(i)}, k={k}: align their sizes first"
            )
        if self.score == "cosine":
            q_i = F.normalize(q_i, dim=-1)
            key_i = F.normalize(key_i, dim=-1)
            a = batched_tensor_product(q_i, key_i, (1, 2, 3), (1, 2, 3), 3)
            a = self._apply_temperature(a, q, i, k, head_mode)
        else:
            a = batched_tensor_product(q_i, key_i, (1, 2, 3), (1, 2, 3), 3) / math.sqrt(q_i.size(-1))
        a = self.attn_drop(a.softmax(dim=-1))

        # A x^(1,2,3)_(1,3,2) V[i;k], whose two transposes cancel
        z = a @ value_i
        return batched_fold(z, i, k, q.shape)


class BilinearAlign(nn.Module):
    """Resizes the memory to the size of x, for forms whose J3 depends on the spatial size."""

    def __init__(self, dim=None):
        super().__init__()

    def forward(self, memory, x):
        if memory.shape[-2:] == x.shape[-2:]:
            return memory
        return F.interpolate(memory, x.shape[-2:], mode="bilinear")


def _form(form):
    """A form of attention is the pair (i, k): reordering vector and mode, both given explicitly."""
    try:
        i, k = form
        i, k = tuple(int(n) for n in i), int(k)
    except (TypeError, ValueError):
        raise ValueError(f"a form of attention is (i, k), e.g. ((1, 2, 3, 4, 5), 5); got {form!r}") from None
    if len(i) not in (4, 5):
        raise ValueError(f"i has 5 modes (B x d x P x S x C) or 4 (B x d x P x SC); got {i}")
    _unfolding_layout(i, k, len(i))  # checks i and k
    return i, k


def _uses_blocks(i, k):
    """CAL-SA (J3 = S) relates channels over the whole image. The forms that relate positions or blocks are blocked."""
    return not (len(i) == 5 and i[k - 1] == 4)


def _j3_depends_on_size(i, k):
    """J3 is C for Spa-SA. For the other forms it holds the spatial size, so Q and K must be aligned."""
    return not (len(i) == 5 and i[k - 1] == 5)


def _to_blocks(x, q):
    """[B, C, H, W] -> [B, P, q, q, C], with one block of the whole image when q is None."""
    return x.movedim(1, -1).unsqueeze(1) if q is None else tensor_blocking(x, q)


def _from_blocks(t, q, h, w):
    return t.squeeze(1).movedim(-1, 1) if q is None else tensor_unblocking(t, h, w)


def _to_factor(blocks, i, num_heads):
    """[B, P, q, q, d * C] -> B x d x P x S x C, or B x d x P x SC when i has four modes."""
    b, p, q_h, q_w, channels = blocks.shape
    factor = blocks.reshape(b, p, q_h * q_w, num_heads, channels // num_heads).permute(0, 3, 1, 2, 4)
    return factor.flatten(-2) if len(i) == 4 else factor


def _from_factor(factor, i, h, w):
    """Inverse of `_to_factor`: B x d x P x S x C (or B x d x P x SC) -> [B, P, h, w, d * C]."""
    if len(i) == 4:
        factor = factor.unflatten(-1, (h * w, -1))
    b, d, p, _, c = factor.shape
    return factor.permute(0, 2, 3, 1, 4).reshape(b, p, h, w, d * c)


class InScaleAttention(nn.Module):
    """Attention inside a branch, `GeneralizedAttention` with the form (i, k).

    W_Q, W_K, W_V are the 1 x 1 convolutions of Section III-B, so they act on the channels. For
    CAL-SA J3 = S is the spatial size, and acting on the channels lets one layer take images of any size.
    The CAL-SA form is adapted from MDTA in Restormer (see THIRD_PARTY_NOTICES.md).
    """

    def __init__(self, dim, num_heads, form, branch, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.i, self.k = _form(form)
        self.branch = branch
        self.window = None  # set by FCFormer.set_window_size
        self.w_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.w_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.w_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.attention = GeneralizedAttention(num_heads, attn_drop=attn_drop)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def set_windows(self, branch_window):
        self.window = branch_window[self.branch] if _uses_blocks(self.i, self.k) else None

    def forward(self, x):
        h, w = x.shape[-2:]
        q_h, q_w = (h, w) if self.window is None else (self.window, self.window)
        # Stored separately, computed in one call, so outputs match training element by element.
        # Three separate calls give the same formula but can round differently by about one float unit.
        weight = torch.cat([self.w_q.weight, self.w_k.weight, self.w_v.weight])
        q, k, v = (
            _to_factor(_to_blocks(f, self.window), self.i, self.num_heads)
            for f in F.conv2d(x, weight).chunk(3, dim=1)
        )
        out = _from_factor(self.attention(q, k, v, i=self.i, k=self.k), self.i, q_h, q_w)
        return self.project_out(_from_blocks(out, self.window, h, w))


class CrossScaleAttention(nn.Module):
    """Attention between two branches, `GeneralizedAttention` with the form (i, k).

    Q comes from the branch being updated and K, V from the other one (Section III-B). Blocking uses
    a window per branch (16 / 8 / 4 for HR / MR / LR, Table VII). All branches then have the same
    number of blocks, so block p of Q attends to block p of K and V across resolutions.

    W_Q, W_K, W_V are linear layers with bias on the channels. With Spa-SA, Q and K may differ in
    their spatial size. Forms whose J3 depends on it (CAL-SA, Patch-SA) need `align`, a module
    `align(memory, x)` that brings the memory to the size of x. Spa-SA does not use it.
    """

    def __init__(self, dim, memory_dim, num_heads, form, branch, memory_branch, align=None, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.i, self.k = _form(form)
        self.branch = branch
        self.memory_branch = memory_branch
        self.query_window = None  # set by FCFormer.set_window_size
        self.memory_window = None
        self.memory_proj = nn.Conv2d(memory_dim, dim, kernel_size=1, stride=1)
        self.align = align(dim) if align is not None and _j3_depends_on_size(self.i, self.k) else None
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.attention = GeneralizedAttention(num_heads, attn_drop=attn_drop)
        self.project_out = nn.Linear(dim, dim)
        self.local_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False)
        self.local_act = nn.LeakyReLU(0.2)
        self.local_pwconv = nn.Conv2d(dim, dim, 1, groups=dim)

    def set_windows(self, branch_window):
        if not _uses_blocks(self.i, self.k):
            self.query_window = self.memory_window = None
        else:
            self.query_window = branch_window[self.branch]
            aligned = self.align is not None
            self.memory_window = self.query_window if aligned else branch_window[self.memory_branch]

    def local_path(self, x, memory):
        """The local part of the response: the attention above relates positions inside windows,
        and this path combines x and the resized memory position by position, with a depthwise
        3 x 3 convolution, a gate (one half times the other half) and a 1 x 1 convolution.
        It is added to the attention output, and is part of the published weights.
        """
        memory = F.interpolate(memory, x.shape[-2:], mode="bilinear")
        gate, value = self.local_dwconv(torch.cat([x, memory], dim=1)).chunk(2, dim=1)
        return self.local_pwconv(self.local_act(gate * value))

    def forward(self, x, memory):
        h, w = x.shape[-2:]
        q_h, q_w = (h, w) if self.query_window is None else (self.query_window, self.query_window)
        memory = self.memory_proj(memory)
        if self.align is not None:
            memory = self.align(memory, x)

        # Definition 1
        x_blocks = _to_blocks(x, self.query_window)
        memory_blocks = _to_blocks(memory, self.memory_window)

        # The projections act on the channels, the last mode after blocking.
        # W_K and W_V in one call, for the same reason as in InScaleAttention.
        weight = torch.cat([self.w_k.weight, self.w_v.weight])
        bias = torch.cat([self.w_k.bias, self.w_v.bias])
        q = self.w_q(x_blocks)
        k, v = F.linear(memory_blocks, weight, bias).chunk(2, dim=-1)
        q, k, v = (_to_factor(f, self.i, self.num_heads) for f in (q, k, v))

        out = _from_factor(self.attention(q, k, v, i=self.i, k=self.k), self.i, q_h, q_w)
        return _from_blocks(self.project_out(out), self.query_window, h, w) + self.local_path(x, memory)
