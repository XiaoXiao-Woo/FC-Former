"""FC-Former for pansharpening in PanCollection: data keys ms / lms / pan / gt, output I_f = net + lms."""
import torch
import torch.nn.functional as F

from fcformer.model.fcformer import FCFormer
from fcformer.tasks._config import device_from_cfg, load_pretrained, network_kwargs
from fcformer.tasks.losses import SupervisedLoss


class FCFormerPansharpening(FCFormer):
    """I_1 = pan, I_H = lms, I_M = ms upsampled to half of lms, I_L = ms."""

    def _fuse(self, data):
        ms, lms, pan = data["ms"], data["lms"], data["pan"]
        mms = F.interpolate(
            ms, size=(lms.size(-2) // 2, lms.size(-1) // 2), mode="bilinear", align_corners=True
        )
        return FCFormer.forward(self, pan, lms, mms, ms) + lms

    def forward(self, data, mode, **kwargs):
        """Entry point used by the udl_vis trainer."""
        if "train" in mode:
            return self.train_step(data, **kwargs)
        return self.val_step(data, **kwargs)

    def train_step(self, data, *args, **kwargs):
        return self.criterion({"sr": self._fuse(data)}, data["gt"])

    @torch.no_grad()
    def val_step(self, data, **kwargs):
        return self._fuse(data)


def _spectral_num_from_cfg(cfg):
    for key in ("train_name", "val_name", "test_name"):
        if hasattr(cfg.dataset, key):
            name = getattr(cfg.dataset, key)
            return 8 if name and "wv" in str(name) else 4
    return int(getattr(cfg, "hs_channel", 8) or 8)


try:
    from pancollection.models.base_model import PanSharpeningModel
    from udl_vis.Basis.optim import get_optimizer
except ImportError:  # the network and this adapter do not need PanCollection
    PanSharpeningModel = None

if PanSharpeningModel is not None:

    class build_FCFormer(PanSharpeningModel, name="FCFormer"):
        def __call__(self, cfg, logger=None):
            # Eq. (11): L = L1 + 0.1 L_SSIM.
            device = device_from_cfg(cfg)
            criterion = SupervisedLoss(ssim_weight=0.1)
            losses = {"loss": criterion}
            model = FCFormerPansharpening(
                spectral_num=_spectral_num_from_cfg(cfg),
                guide_channels=int(getattr(cfg, "added_c", 1) or 1),
                **network_kwargs(cfg),
            ).to(device)
            model.device = device
            model.criterion = criterion
            load_pretrained(model, cfg)
            optimizer = get_optimizer(model, model.parameters(), **cfg.optimizer_cfg)
            return model, losses, optimizer, None
