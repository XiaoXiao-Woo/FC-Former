"""FC-Former for multispectral and hyperspectral image fusion (MHIF), in HyperSpectralCollection (hisr).

Data keys lrhsi / up / rgb / gt, output I_f = net + up.
"""
import torch
import torch.nn.functional as F

from fcformer.model.fcformer import FCFormer
from fcformer.tasks._config import device_from_cfg, flag_from_cfg, load_pretrained, network_kwargs
from fcformer.tasks.losses import SupervisedLoss


class FCFormerHyperspectral(FCFormer):
    """I_1 = rgb (HR-MSI), I_H = up (LR-HSI upsampled), I_M = lrhsi upsampled to half of up, I_L = lrhsi.

    flip_guide: reverse the channel order of rgb before use. The released CAVE weights take the RGB
        channels in the reverse order of the files in elsting/MHIF on Hugging Face.
    """

    def __init__(self, spectral_num=31, guide_channels=3, flip_guide=False, **kwargs):
        super().__init__(spectral_num, guide_channels, **kwargs)
        self.flip_guide = flip_guide

    def _fuse(self, data):
        lrhsi, up, rgb = data["lrhsi"], data["up"], data["rgb"]
        if self.flip_guide:
            rgb = rgb.flip(1)
        mid = F.interpolate(
            lrhsi, size=(up.size(-2) // 2, up.size(-1) // 2), mode="bilinear", align_corners=True
        )
        return FCFormer.forward(self, rgb, up, mid, lrhsi) + up

    def forward(self, data, mode, **kwargs):
        """Entry point used by the udl_vis trainer."""
        if "train" in mode:
            return self.train_step(data, **kwargs)
        return self.val_step(data, **kwargs)

    def train_step(self, data, *args, **kwargs):
        """`criterion` is Eq. (11), L1 + 0.1 L_SSIM (fcformer.tasks.losses)."""
        return self.criterion({"sr": self._fuse(data)}, data["gt"])

    @torch.no_grad()
    def val_step(self, data, **kwargs):
        return self._fuse(data)


try:
    from hisr.models.base_model import HISRModel
    from udl_vis.Basis.optim import get_optimizer
except ImportError:  # the network and this adapter do not need hisr
    HISRModel = None

if HISRModel is not None:

    class build_FCFormer(HISRModel, name="FCFormer"):
        def __call__(self, cfg, logger=None):
            # Eq. (11): L = L1 + 0.1 L_SSIM.
            device = device_from_cfg(cfg)
            criterion = SupervisedLoss(ssim_weight=0.1)
            model = FCFormerHyperspectral(
                spectral_num=int(getattr(cfg, "spectral_num", 31) or 31),
                guide_channels=int(getattr(cfg, "guide_channels", 3) or 3),
                flip_guide=flag_from_cfg(cfg, "flip_guide"),
                scale=int(getattr(cfg, "scale_factor", 4) or 4),
                **network_kwargs(cfg),
            ).to(device)
            model.device = device
            model.criterion = criterion
            load_pretrained(model, cfg)
            optimizer = get_optimizer(model, model.parameters(), **cfg.optimizer_cfg)
            return model, criterion, optimizer, None
