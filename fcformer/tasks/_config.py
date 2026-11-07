"""Reading a udl_vis config into FC-Former: network arguments, device and weights. Used by the task builders.

udl_vis passes command-line values such as `null`, `none` or `false` on as strings; `_unset` treats them
as not set.
"""
import os

import torch

from fcformer.model.generalized_attention import BilinearAlign

_ALIGN = {"bilinear": BilinearAlign}


def _unset(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none", "false", "~"))


def _get(cfg, name, default=None):
    value = getattr(cfg, name, default)
    return default if _unset(value) else value


def device_from_cfg(cfg):
    """`device` from the config, else cuda when available, else cpu."""
    return torch.device(_get(cfg, "device") or ("cuda" if torch.cuda.is_available() else "cpu"))


def forms_from_cfg(cfg, name):
    """Forms (i, k) from a config: a list of [i, k]. Functions per stage are for code, not for configs."""
    value = _get(cfg, name)
    return {} if value is None else {name: tuple((tuple(i), int(k)) for i, k in value)}


def flag_from_cfg(cfg, name, default=False):
    """A boolean from the config."""
    value = getattr(cfg, name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def align_from_cfg(cfg):
    """`align` from the config: unset, or the name of an alignment module ("bilinear")."""
    value = _get(cfg, "align")
    if value is None:
        return None
    key = str(value).strip().lower()
    if key not in _ALIGN:
        raise ValueError(f"unknown align {value!r}; expected one of {sorted(_ALIGN)} or null")
    return _ALIGN[key]


def network_kwargs(cfg):
    """Arguments of `FCFormer` that a config may set."""
    return dict(
        channels=_get(cfg, "channel_list", [48, [48, 96], [48, 96, 192]]),
        num_heads=_get(cfg, "num_heads", [8, [8, 8], [8, 8, 8]]),
        mlp_ratio=_get(cfg, "mlp_ratio", [2, [2, 2], [2, 2, 2]]),
        num_blocks=_get(cfg, "block_list", [1, [1, 1], [1, 1, 1]]),
        attn_drop=float(_get(cfg, "attn_drop", 0.0)),
        drop_path=float(_get(cfg, "drop_path", 0.0)),
        mlp_drop=float(_get(cfg, "mlp_drop", 0.0)),
        align=align_from_cfg(cfg),
        **forms_from_cfg(cfg, "in_scale"),
        **forms_from_cfg(cfg, "cross_scale"),
    )


def load_weights(path):
    """A .pth state dict, or a checkpoint saved by training (model.safetensors, or its directory)."""
    path = str(path)
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_pretrained(model, cfg):
    """Loads `pretrained` from the config, if set."""
    pretrained = _get(cfg, "pretrained") or _get(cfg, "pretrained_model")
    if pretrained:
        model.load_state_dict(load_weights(pretrained))
