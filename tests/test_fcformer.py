"""CPU checks that need no data or weights: python -m pytest tests"""
import pytest
import torch
import torch.nn.functional as F

from fcformer import FCFormer
from fcformer.model.generalized_attention import (
    BilinearAlign,
    GeneralizedAttention,
    batched_fold,
    batched_unfold,
    tensor_blocking,
    tensor_unblocking,
)
from fcformer.tasks.hyperspectral import FCFormerHyperspectral
from fcformer.tasks.pansharpening import FCFormerPansharpening


def test_tensor_blocking_round_trip():
    x = torch.rand(2, 6, 32, 16)
    assert tensor_blocking(x, 8).shape == (2, 4 * 2, 8, 8, 6)
    assert torch.equal(tensor_unblocking(tensor_blocking(x, 8), 32, 16), x)


@pytest.mark.parametrize("i, k, shape", [
    ((1, 2, 3, 4), None, (2, 4, 6, 9)),
    ((1, 2, 4, 3), None, (2, 4, 6, 9)),
    ((1, 2, 3, 5, 4), None, (2, 3, 4, 6, 9)),
    ((1, 2, 3, 4, 5), 3, (2, 4, 6, 3, 3)),
    ((2, 1, 3, 4), 4, (4, 2, 6, 9)),
])
def test_unfold_and_fold_are_inverse(i, k, shape):
    x = torch.rand(*shape)
    k = k or len(i)
    assert torch.equal(batched_fold(batched_unfold(x, i, k), i, k, shape), x)


def test_unfolding_sizes():
    x = torch.rand(2, 4, 6, 9)
    assert batched_unfold(x, (1, 2, 3, 4), 4).shape == (8, 6, 9)
    assert batched_unfold(x, (1, 2, 4, 3), 4).shape == (8, 9, 6)
    assert batched_unfold(torch.rand(2, 4, 6, 5, 7), (1, 2, 3, 4, 5), 3).shape == (8, 35, 6)


def _explicit(q, k, v, temperature):
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    a = (q @ k.transpose(-2, -1)) * temperature
    return a.softmax(dim=-1) @ v


def test_channel_form_matches_explicit_formula():
    # B x d x C x S, i = (1, 2, 3, 4), k = 4: the channel example of Section III-A
    q, k, v = torch.rand(2, 4, 6, 9), torch.rand(2, 4, 6, 9), torch.rand(2, 4, 6, 9)
    attention = GeneralizedAttention(num_heads=4)
    attention.temperature.data = torch.rand(4, 1, 1)
    assert torch.allclose(attention(q, k, v, i=(1, 2, 3, 4), k=4), _explicit(q, k, v, attention.temperature))


def test_spatial_form_matches_explicit_formula_across_sizes():
    # Spa-SA on B x d x P x S x C, with fewer positions in K and V than in Q
    q, k, v = torch.rand(2, 4, 3, 16, 6), torch.rand(2, 4, 3, 9, 6), torch.rand(2, 4, 3, 9, 6)
    attention = GeneralizedAttention(num_heads=4)
    attention.temperature.data = torch.rand(4, 1, 1)
    expected = _explicit(q, k, v, attention.temperature.view(4, 1, 1, 1))
    assert torch.allclose(attention(q, k, v, i=(1, 2, 3, 4, 5), k=5), expected)


def test_channel_form_on_blocks_matches_explicit_formula():
    # CAL-SA on B x d x P x S x C: A is C x C
    q, k, v = torch.rand(2, 4, 3, 16, 6), torch.rand(2, 4, 3, 16, 6), torch.rand(2, 4, 3, 16, 6)
    attention = GeneralizedAttention(num_heads=4)
    attention.temperature.data = torch.rand(4, 1, 1)
    t = lambda x: x.transpose(-2, -1)
    expected = t(_explicit(t(q), t(k), t(v), attention.temperature.view(4, 1, 1, 1)))
    assert torch.allclose(attention(q, k, v, i=(1, 2, 3, 5, 4), k=5), expected)


def test_mode_below_the_last():
    # B x d x C x H x W with k = 3: J1 = Bd, J2 = HW, J3 = C
    q, k, v = torch.rand(2, 4, 6, 3, 5), torch.rand(2, 4, 6, 3, 5), torch.rand(2, 4, 6, 3, 5)
    attention = GeneralizedAttention(num_heads=4)
    out = attention(q, k, v, i=(1, 2, 3, 4, 5), k=3)
    flat = lambda t: t.flatten(-2).transpose(-2, -1)
    expected = _explicit(flat(q), flat(k), flat(v), attention.temperature).transpose(-2, -1).reshape(q.shape)
    assert out.shape == q.shape
    assert torch.allclose(out, expected)


def test_scaled_dot_is_the_formula_of_the_theorem():
    q, k, v = torch.rand(2, 4, 6, 9), torch.rand(2, 4, 6, 9), torch.rand(2, 4, 6, 9)
    expected = ((q @ k.transpose(-2, -1)) / 3).softmax(dim=-1) @ v
    assert torch.allclose(GeneralizedAttention(score="scaled_dot")(q, k, v, i=(1, 2, 3, 4), k=4), expected)


def test_head_mode_must_be_in_the_batch_part():
    q = torch.rand(2, 4, 6, 9)
    with pytest.raises(ValueError):
        GeneralizedAttention(num_heads=4)(q, q, q, i=(1, 2, 3, 4), k=4, head_mode=3)


@pytest.mark.parametrize("size", [64, 128])
def test_pansharpening_shape(size):
    model = FCFormerPansharpening(spectral_num=8).eval()
    data = {
        "ms": torch.rand(1, 8, size // 4, size // 4),
        "lms": torch.rand(1, 8, size, size),
        "pan": torch.rand(1, 1, size, size),
    }
    assert model.val_step(data).shape == (1, 8, size, size)


def test_scale_8_network():
    model = FCFormer(spectral_num=31, guide_channels=3, scale=8).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 128, 128), torch.rand(1, 31, 128, 128),
                    torch.rand(1, 31, 64, 64), torch.rand(1, 31, 16, 16))
    assert out.shape == (1, 31, 128, 128)


def test_parameter_count_matches_table_2():
    # 3.75M in Table II of the paper (31 bands, 3-band guide)
    model = FCFormer(spectral_num=31, guide_channels=3)
    assert round(sum(p.numel() for p in model.parameters()) / 1e6, 2) == 3.75


# Forms (i, k) on B x d x P x S x C (or B x d x P x SC), the order of Section III-A
SPA, CAL, PATCH = ((1, 2, 3, 4, 5), 5), ((1, 2, 3, 5, 4), 5), ((1, 2, 3, 4), 4)


@pytest.mark.parametrize("config", [
    dict(),                                                  # Table VI row VI
    dict(in_scale=()),                                       # row III
    dict(in_scale=(SPA,)),                                   # row IV
    dict(in_scale=(PATCH,)),                                 # Patch-SA in a branch
    dict(in_scale=(), cross_scale=(CAL,), align=BilinearAlign),   # row II
    dict(cross_scale=(PATCH,), align=BilinearAlign),         # Patch-SA between branches
    dict(in_scale=(CAL, SPA)),                               # two Z_k, Algorithm 1
    dict(in_scale=(SPA, PATCH), cross_scale=(SPA, CAL), align=BilinearAlign),
])
def test_forms_are_selected_by_the_reordering_vectors(config):
    model = FCFormerPansharpening(spectral_num=4, **config).eval()
    data = {"ms": torch.rand(1, 4, 16, 16), "lms": torch.rand(1, 4, 64, 64), "pan": torch.rand(1, 1, 64, 64)}
    assert model.val_step(data).shape == (1, 4, 64, 64)


def test_patch_form_matches_explicit_formula():
    # B x d x P x SC, attending over the P blocks
    q, k, v = torch.rand(2, 4, 5, 12), torch.rand(2, 4, 5, 12), torch.rand(2, 4, 5, 12)
    attention = GeneralizedAttention(num_heads=4)
    attention.temperature.data = torch.rand(4, 1, 1)
    expected = _explicit(q, k, v, attention.temperature)
    assert torch.allclose(attention(q, k, v, i=PATCH[0], k=PATCH[1]), expected)


def test_channel_form_across_scales_needs_align():
    with pytest.raises(ValueError, match="align"):
        model = FCFormerPansharpening(spectral_num=4, cross_scale=(CAL,)).eval()
        model.val_step({"ms": torch.rand(1, 4, 16, 16), "lms": torch.rand(1, 4, 64, 64), "pan": torch.rand(1, 1, 64, 64)})


def test_cross_scale_needs_a_vector():
    with pytest.raises(ValueError, match="cross_scale"):
        FCFormer(spectral_num=4, cross_scale=())


def _vectors(model):
    return {
        name: (module.i, module.k) for name, module in model.named_modules()
        if type(module).__name__ in ("InScaleAttention", "CrossScaleAttention")
    }


def test_vectors_per_stage_and_branch():
    def in_scale(stage, branch):
        return (SPA,) if stage == 3 else (CAL,)

    def cross_scale(stage, branch, memory_branch):
        return (CAL,) if branch == 2 and memory_branch == 1 else (SPA,)

    model = FCFormerPansharpening(spectral_num=4, in_scale=in_scale, cross_scale=cross_scale, align=BilinearAlign).eval()
    vectors = _vectors(model)
    assert vectors["stage3.in_scale.0.0.attn"] == SPA
    assert vectors["stage2.in_scale.0.0.attn"] == CAL
    assert vectors["stage1_hr_in_scale.attn"] == CAL
    assert vectors["stage2.low_branch_msir.fusion.in_scale.attn"] == CAL
    assert vectors["stage3.cross_scale.2_from_1.attn"] == CAL
    assert vectors["stage2.low_branch_msir.cross_from_mr.attn"] == CAL
    assert vectors["stage3.cross_scale.2_from_0.attn"] == SPA
    data = {"ms": torch.rand(1, 4, 16, 16), "lms": torch.rand(1, 4, 64, 64), "pan": torch.rand(1, 1, 64, 64)}
    assert model.val_step(data).shape == (1, 4, 64, 64)


def test_no_in_scale_attention_in_one_branch():
    model = FCFormerPansharpening(spectral_num=4, in_scale=lambda stage, branch: () if branch == 2 else (CAL,))
    assert "stage3.in_scale.2.0.attn" not in _vectors(model)
    assert "stage3.in_scale.1.0.attn" in _vectors(model)


def test_a_missing_cross_scale_vector_is_reported():
    with pytest.raises(ValueError, match="cross_scale"):
        FCFormer(spectral_num=4, cross_scale=lambda stage, branch, memory_branch: () if stage == 3 else (SPA,))


def test_align_is_used_only_by_forms_that_need_it():
    model = FCFormer(spectral_num=4, cross_scale=(SPA, CAL), align=BilinearAlign)
    sum_of_two = model.stage3.cross_scale["0_from_1"].attn
    assert sum_of_two[0].align is None
    assert sum_of_two[1].align is not None


def test_a_form_needs_its_mode():
    with pytest.raises(ValueError, match=r"\(i, k\)"):
        FCFormer(spectral_num=4, in_scale=((1, 2, 3, 5, 4),))


def test_hyperspectral_output_is_the_network_plus_up():
    model = FCFormerHyperspectral().eval()
    data = {"lrhsi": torch.rand(1, 31, 16, 16), "up": torch.rand(1, 31, 64, 64), "rgb": torch.rand(1, 3, 64, 64)}
    mid = F.interpolate(data["lrhsi"], size=(32, 32), mode="bilinear", align_corners=True)
    with torch.no_grad():
        expected = FCFormer.forward(model, data["rgb"], data["up"], mid, data["lrhsi"]) + data["up"]
    assert torch.equal(model.val_step(data), expected)


def test_hyperspectral_train_step():
    model = FCFormerHyperspectral()
    model.criterion = lambda out, gt: (out["sr"] - gt).abs().mean()
    data = {
        "lrhsi": torch.rand(1, 31, 16, 16), "up": torch.rand(1, 31, 64, 64),
        "rgb": torch.rand(1, 3, 64, 64), "gt": torch.rand(1, 31, 64, 64),
    }
    model(data, "train").backward()
    assert all(p.grad is not None for p in model.parameters())


def test_scale_8_fits_the_window_to_the_image_size():
    model = FCFormer(spectral_num=1, guide_channels=1, scale=8, channels=(8, (8, 16), (8, 16, 24)),
                     num_heads=(4, (4, 4), (4, 4, 4)), mlp_ratio=(1, (1, 1), (1, 1, 1)))
    model._fit_window_size(1000)
    assert model.stage3.cross_scale["2_from_0"].attn.query_window == 5
    model._fit_window_size(128)
    assert model.stage3.cross_scale["2_from_0"].attn.query_window == 2


def test_size_error_names_the_blocks():
    model = FCFormerPansharpening(spectral_num=8).eval()
    data = {"ms": torch.rand(1, 8, 18, 18), "lms": torch.rand(1, 8, 72, 72), "pan": torch.rand(1, 1, 72, 72)}
    with pytest.raises(ValueError, match="blocks"):
        model.val_step(data)


def test_supervised_loss_is_eq_11():
    from fcformer.tasks.losses import SupervisedLoss, ssim
    g = torch.Generator().manual_seed(0)
    sr, gt = torch.rand(2, 4, 32, 32, generator=g), torch.rand(2, 4, 32, 32, generator=g)
    out = SupervisedLoss()({"sr": sr}, gt)
    assert torch.allclose(out["loss"], F.l1_loss(sr, gt) + 0.1 * (1 - ssim(sr, gt)))
    assert torch.allclose(ssim(gt, gt), torch.tensor(1.0))

