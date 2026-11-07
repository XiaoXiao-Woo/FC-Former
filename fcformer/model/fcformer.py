"""FC-Former: the network of Section III-E, Fig. 6 and Algorithm 2 of the paper.

    X. Wu, Z.-H. Cao, T.-Z. Huang, L.-J. Deng, J. Chanussot, G. Vivone,
    "Fully-Connected Transformer for Multi-Source Image Fusion", IEEE TPAMI 47(3), 2025.

Notation of the paper: I_1 is the high-resolution source (PAN, HR-MSI, infrared, ...) and I_2 the
source to be sharpened (MS, LR-HSI, ...). I_H, I_M, I_L are I_2 at the HR, MR and LR branch
resolutions. F_H, F_M, F_L are the branch features. Algorithms 1 and 2 give the general scheme, and
the classes below state the concrete choices of FC-Former.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from fcformer.model.generalized_attention import CrossScaleAttention, InScaleAttention

# Used wherever weights are normalized: w / (sum(w) + EPSILON)
EPSILON = 1e-4


class BiasFreeLayerNorm(nn.Module):
    """Layer norm over channels of a [B, C, H, W] tensor, without centering and without bias.

    Adapted from Restormer (see THIRD_PARTY_NOTICES.md).
    """

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(sigma + 1e-5) * self.weight
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class GatedFeedForward(nn.Module):
    """The feed-forward part of a transformer layer: a gated depthwise convolution.

    Adapted from GDFN in Restormer (see THIRD_PARTY_NOTICES.md).
    """

    def __init__(self, dim, expansion):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        for conv in (self.project_in, self.dwconv, self.project_out):
            nn.init.xavier_normal_(conv.weight)

    def forward(self, x):
        gate, value = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(gate) * value)


class TransformerLayer(nn.Module):
    """The layer around the attention: Z = X + LN(project(AV)), then Z = Z + FFN(LN(Z)).

    Eq. (1) and (5) write the core as Z = AV + X. In practice A x V first goes through the attention's
    `project_out` and a layer norm, and a feed-forward step follows, as in a usual transformer block.

    With several forms (i, k), their responses are summed, Z = sum_k Z_k (line 7 of Algorithm 1).
    Eq. (5) writes the residual inside each Z_k; here the attention responses are summed first and the
    residual X is added once. One form is kept as the module itself, several as a list.
    """

    def __init__(self, attentions, dim, mlp_ratio, drop_path=0.0):
        super().__init__()
        self.attn = attentions[0] if len(attentions) == 1 else nn.ModuleList(attentions)
        self.norm1 = BiasFreeLayerNorm(dim)
        self.norm2 = BiasFreeLayerNorm(dim)
        self.ffn = GatedFeedForward(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attention(self, x, *memory):
        if not isinstance(self.attn, nn.ModuleList):
            return self.attn(x, *memory)
        z = self.attn[0](x, *memory)
        for attn in self.attn[1:]:
            z = z + attn(x, *memory)
        return z

    def forward(self, x, *memory):
        x = x + self.drop_path(self.norm1(self.attention(x, *memory)))
        return x + self.drop_path(self.ffn(self.norm2(x)))


def in_scale_layer(dim, num_heads, mlp_ratio, forms, branch, attn_drop=0.0, drop_path=0.0):
    """A layer of in-scale attention, one Z_k per form (i, k). No forms: no layer."""
    if not forms:
        return nn.Identity()
    attentions = [InScaleAttention(dim, num_heads, form, branch, attn_drop) for form in forms]
    return TransformerLayer(attentions, dim, mlp_ratio, drop_path)


def cross_scale_layer(dim, memory_dim, num_heads, mlp_ratio, forms, branch, memory_branch, align=None,
                      attn_drop=0.0, drop_path=0.0):
    """A layer of cross-scale attention, one Z_k per form (i, k)."""
    if not forms:
        raise ValueError(f"cross_scale needs at least one form (i, k), none for branch {branch} from {memory_branch}")
    attentions = [
        CrossScaleAttention(dim, memory_dim, num_heads, form, branch, memory_branch, align, attn_drop)
        for form in forms
    ]
    return TransformerLayer(attentions, dim, mlp_ratio, drop_path)


class Stem(nn.Module):
    """Head of the HR branch in Fig. 6: embed [I_1, I_H].

    The MSIR at this head is a channel concatenation (in `FCFormer.forward`), followed by these convolutions.
    """

    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.norm1 = BiasFreeLayerNorm(dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        )
        self.norm2 = BiasFreeLayerNorm(dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        return self.relu(self.norm2(self.conv2(x)))


class Bottleneck(nn.Module):
    """The Bottleneck box of Fig. 6, adapted from HRNet [34] (see THIRD_PARTY_NOTICES.md).

    Implementation details that differ from the box drawn in Fig. 6: the middle 3 x 3 convolution is a
    1 x 1 convolution followed by a depthwise 3 x 3 one, the activations are LeakyReLU (0.2), and no
    activation follows the residual addition. The published weights use this form.
    """

    expansion = 4

    def __init__(self, in_channels, dim, shortcut=None):
        super().__init__()
        out_channels = dim * self.expansion
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=1, bias=False)
        self.norm1 = BiasFreeLayerNorm(dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        )
        self.norm2 = BiasFreeLayerNorm(dim)
        self.conv3 = nn.Conv2d(dim, out_channels, kernel_size=1, bias=False)
        self.norm3 = BiasFreeLayerNorm(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return out + (x if self.shortcut is None else self.shortcut(x))


def source_embedding(source_channels, dim):
    """Lift the source image I_2 to the channels of a branch."""
    return nn.Sequential(
        nn.Conv2d(source_channels, dim, 1), nn.Conv2d(dim, dim, 3, padding=1, bias=False)
    )


class DynamicBranchFusion(nn.Module):
    """DBF of Section III-D, the baseline of MSIR: a weighted sum, then in-scale attention.

    The published models use DBF (Table V, first column), and this code implements only DBF, not
    the model-based branch fusion of Eq. (7)-(10). The attention here has a fixed feed-forward expansion of 2.
    """

    def __init__(self, dim, num_heads, in_scale, branch):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2))
        self.relu = nn.LeakyReLU(0.2)
        self.in_scale = in_scale_layer(dim, num_heads, 2, in_scale, branch)

    def forward(self, feature, source):
        w = self.relu(self.weights)
        w = w / (w.sum(dim=0) + EPSILON)
        return self.in_scale(self.relu(w[0] * feature + w[1] * source))


class MediumBranchMSIR(nn.Module):
    """MSIR at the head of the MR branch (Stage 1 of Fig. 6): F_M from F_H and I_M.

    Implementation detail that differs from Fig. 6: Fig. 6 draws Spa-SA from the HR branch into this head.
    Here F_H reaches the MR resolution through a strided convolution, and cross-scale attention is used
    from the next head on (`LowBranchMSIR`). The published weights use this form.
    """

    def __init__(self, in_channels, dim, source_channels, num_heads, in_scale):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim),
            BiasFreeLayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.embed_source = source_embedding(source_channels, dim)
        self.fusion = DynamicBranchFusion(dim, num_heads, in_scale, branch=1)
        self.norm = BiasFreeLayerNorm(dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, f_h, i_m):
        source = self.embed_source(i_m)
        return self.relu(self.norm(self.fusion(self.downsample(f_h), source)))


class LowBranchMSIR(nn.Module):
    """MSIR at the head of the LR branch (Stage 2 of Fig. 6): F_L from F_H, F_M and I_L."""

    def __init__(self, channels, source_channels, num_heads, mlp_ratio, in_scale, cross_from_hr, cross_from_mr,
                 align, attn_drop, drop_path):
        super().__init__()
        c_h, c_m, c_l = channels
        self.embed_source = source_embedding(source_channels, c_l)
        self.cross_from_hr = cross_scale_layer(
            c_l, c_h, num_heads, mlp_ratio, cross_from_hr, 2, 0, align, attn_drop, drop_path
        )
        self.cross_from_mr = cross_scale_layer(
            c_l, c_m, num_heads, mlp_ratio, cross_from_mr, 2, 1, align, attn_drop, drop_path
        )
        self.fusion = DynamicBranchFusion(c_l, num_heads, in_scale, branch=2)
        self.weights = nn.Parameter(torch.ones(2))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, f_h, f_m, i_l):
        w = self.relu(self.weights)
        w = w / (w.sum(dim=0) + EPSILON)
        source = self.embed_source(i_l)
        from_hr = self.cross_from_hr(source, f_h)
        from_mr = self.cross_from_mr(source, f_m)
        return w[0] * from_mr + w[1] * self.fusion(from_hr, source)


class FCSAStage(nn.Module):
    """One stage of the FCSA framework: Eq. (6) and Algorithm 1 (Fig. 5).

    Each branch first runs its in-scale attention. Then every branch receives a weighted sum over all
    branches, using cross-scale attention for the pairs across scales: Z_i = sum_j w_ij * Z_(i<-j).
    With `next_channels`, the stage also opens the next branch through MSIR (Stage 2 of Fig. 6).

    Eq. (6) lists source images among the factors of each branch. In FC-Former they enter at the branch
    heads (Fig. 6), so here Q, K, V come from branch features. The learned weights w_ij of the sum are an
    implementation choice of FC-Former; Algorithm 1 sums without weights.
    """

    def __init__(self, stage, channels, num_blocks, num_heads, mlp_ratio, in_scale, cross_scale, align,
                 attn_drop, drop_path, next_channels=None, source_channels=None):
        super().__init__()
        n = len(channels)
        # `attn_drop` and `drop_path` here are the rates passed by FCFormer, see its docstring.
        self.in_scale = nn.ModuleList(
            nn.Sequential(*[
                in_scale_layer(channels[i], num_heads[i], mlp_ratio[i], in_scale(stage, i), i, attn_drop, drop_path)
                for _ in range(num_blocks[i])
            ])
            for i in range(n)
        )
        self.cross_scale = nn.ModuleDict({
            f"{i}_from_{j}": cross_scale_layer(
                channels[i], channels[j], num_heads[i], mlp_ratio[i], cross_scale(stage, i, j), i, j, align,
                attn_drop=drop_path, drop_path=0.2
            )
            for i in range(n) for j in range(n) if i != j
        })
        self.low_branch_msir = None
        if next_channels is not None:
            self.low_branch_msir = LowBranchMSIR(
                next_channels, source_channels, num_heads[-1], mlp_ratio[-1], in_scale(stage, n),
                cross_scale(stage, n, 0), cross_scale(stage, n, 1), align,
                attn_drop, drop_path
            )
        self.relu = nn.LeakyReLU(0.2)
        # w_ij, a learned weight per pair of branches, normalized over the receiving branches of one
        # source. The extra last column is not used. The extra last row is not read in `forward`,
        # but it enters the normalization.
        self.branch_weights = nn.Parameter(torch.ones(n + 1, n + 1))

    def forward(self, x, i_l=None):
        w = self.relu(self.branch_weights)
        w = w / (w.sum(dim=0) + EPSILON)

        x = [layers(x_i) for layers, x_i in zip(self.in_scale, x)]

        if self.low_branch_msir is not None:
            f_l = self.low_branch_msir(*x, i_l)

        z = []
        for i, x_i in enumerate(x):
            z_i = w[i, 0] * (x_i if i == 0 else self.cross_scale[f"{i}_from_0"](x_i, x[0]))
            for j in range(1, len(x)):
                z_i = z_i + w[i, j] * (x_i if i == j else self.cross_scale[f"{i}_from_{j}"](x_i, x[j]))
            z.append(self.relu(z_i))

        if self.low_branch_msir is not None:
            z.append(f_l)
        return z


class FCFormer(nn.Module):
    """Algorithm 2: forward(I_1, I_H, I_M, I_L) -> residual of the fused image I_f.

    Args:
        spectral_num: channels of I_2 and of the output.
        guide_channels: channels of I_1.
        channels: (stem, (HR, MR), (HR, MR, LR)), feature channels per stage.
        num_blocks, num_heads, mlp_ratio: same nesting as `channels`. `num_blocks[0]` counts the
            Bottlenecks after the first one, so 1 means two.
        attn_drop: attention dropout of the stage 1 CAL-SA.
        drop_path: drop path of the stage 1 CAL-SA; attention dropout of the later CAL-SA layers and
            of the Spa-SA at the LR head.
        mlp_drop: drop path of the later CAL-SA layers and of the Spa-SA at the LR head; attention
            dropout of the Spa-SA in FCSA. The Spa-SA in FCSA has a fixed drop path of 0.2.
            These rates act only in training.
        scale: resolution ratio r = H / h. The LR branch runs at I_2's own resolution.
        window_size: HR window of the blocked forms. MR and LR follow from the resolutions (Table VII).
        in_scale: forms (i, k) of the attention inside a branch, reordering vector and mode of
            Theorem 2 (written for FCSA in Eq. (6)), one Z_k each (Algorithm 1). Empty means no in-scale attention. The default
            ((1, 2, 3, 5, 4), 5) is CAL-SA, Table VI row VI. Or a function (stage, branch) -> forms,
            for different forms per stage or branch.
        cross_scale: forms (i, k) of the attention between branches, at least one. The default
            ((1, 2, 3, 4, 5), 5) is Spa-SA. See generalized_attention.py for the forms.
            Or a function (stage, branch, memory_branch) -> forms. Stages are 1, 2, 3 and branches
            0, 1, 2 for HR, MR, LR. The LR head belongs to stage 2.
        align: class of a module `align(memory, x)`, built with the channels of the branch, for
            cross-scale forms whose J3 depends on the spatial size. See `BilinearAlign`.
    """

    def __init__(
        self,
        spectral_num,
        guide_channels=1,
        channels=(48, (48, 96), (48, 96, 192)),
        num_blocks=(1, (1, 1), (1, 1, 1)),
        num_heads=(8, (8, 8), (8, 8, 8)),
        mlp_ratio=(2, (2, 2), (2, 2, 2)),
        attn_drop=0.0,
        drop_path=0.0,
        mlp_drop=0.0,
        scale=4,
        window_size=16,
        in_scale=(((1, 2, 3, 5, 4), 5),),
        cross_scale=(((1, 2, 3, 4, 5), 5),),
        align=None,
    ):
        super().__init__()
        self.scale = scale
        in_scale_at = in_scale if callable(in_scale) else lambda stage, branch: in_scale
        cross_scale_at = cross_scale if callable(cross_scale) else lambda stage, branch, memory_branch: cross_scale
        c_stem, c_stage2, c_stage3 = channels
        if tuple(c_stage2) != tuple(c_stage3[:2]):
            raise ValueError("the HR and MR branches must keep their channels from stage 2 to stage 3")

        self.stem = Stem(spectral_num + guide_channels, c_stem)
        c_bottleneck = c_stem * Bottleneck.expansion
        shortcut = nn.Sequential(nn.Conv2d(c_stem, c_bottleneck, 1, bias=False), BiasFreeLayerNorm(c_bottleneck))
        self.bottlenecks = nn.Sequential(
            Bottleneck(c_stem, c_stem, shortcut),
            *[Bottleneck(c_bottleneck, c_stem) for _ in range(num_blocks[0])],
        )

        # Stage 1 of Fig. 6: F_H through in-scale attention, F_M through MSIR.
        self.stage1_hr_proj = nn.Conv2d(c_bottleneck, c_stage2[0], 1)
        self.stage1_hr_in_scale = in_scale_layer(
            c_stage2[0], num_heads[1][0], mlp_ratio[1][0], in_scale_at(1, 0), 0, attn_drop, drop_path
        )
        self.stage1_mr_msir = MediumBranchMSIR(c_bottleneck, c_stage2[1], spectral_num, num_heads[1][-1], in_scale_at(1, 1))

        # Stage 2 of Fig. 6: FCSA over two branches, and F_L through MSIR. Rates: see FCSAStage.
        self.stage2 = FCSAStage(
            2, c_stage2, num_blocks[1], num_heads[1], mlp_ratio[1], in_scale_at, cross_scale_at, align,
            attn_drop=drop_path, drop_path=mlp_drop, next_channels=c_stage3, source_channels=spectral_num,
        )

        # Stage n of Fig. 6: FCSA over three branches.
        self.stage3 = FCSAStage(
            3, c_stage3, num_blocks[2], num_heads[2], mlp_ratio[2], in_scale_at, cross_scale_at, align,
            attn_drop=drop_path, drop_path=mlp_drop,
        )

        c_out = sum(c_stage3)
        self.fusion_head = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(c_out, c_out, 1)),
            ("norm", BiasFreeLayerNorm(c_out)),
            ("act", nn.LeakyReLU(0.2)),
            ("conv2", nn.Conv2d(c_out, spectral_num, 1)),
        ]))

        self.set_window_size(window_size)

    def set_window_size(self, window_size):
        """Sets the HR window. MR and LR follow from the resolutions of the branches."""
        self.window_size = window_size
        self._apply_window_size(window_size)

    def _apply_window_size(self, window_size):
        """Blocking windows per branch: proportional to the resolution of the branch."""
        self._active_window_size = window_size
        branch_window = (window_size, window_size // 2, window_size // self.scale)
        for module in self.modules():
            if isinstance(module, (InScaleAttention, CrossScaleAttention)):
                module.set_windows(branch_window)

    def _fit_window_size(self, size):
        """Scale 8 on images of 1000 x 1000 (Harvard) cannot use a window of 16 and uses 40 / 20 / 5."""
        window_size = self.window_size
        if self.scale == 8 and size % window_size != 0 and size % 40 == 0:
            window_size = 40
        if window_size != self._active_window_size:
            self._apply_window_size(window_size)

    def forward(self, i_1, i_h, i_m, i_l):
        self._fit_window_size(i_1.size(-1))
        x = self.bottlenecks(self.stem(torch.cat([i_1, i_h], dim=1)))

        f_h = self.stage1_hr_in_scale(self.stage1_hr_proj(x))
        f_m = self.stage1_mr_msir(x, i_m)

        z = self.stage2([f_h, f_m], i_l)
        z = self.stage3(z)

        # Channel concat and the two convolutions at the right of Fig. 6
        size = z[0].shape[-2:]
        z_m = F.interpolate(z[1], size=size, mode="bilinear", align_corners=True)
        z_l = F.interpolate(z[2], size=size, mode="bilinear", align_corners=True)
        return self.fusion_head(torch.cat([z[0], z_m, z_l], dim=1))
