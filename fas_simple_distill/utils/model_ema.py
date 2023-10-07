import torch
import torch.nn as nn
from timm.utils.model_ema import ModelEmaV2


class ModelEmaV3(ModelEmaV2):
    def _update(self, model: nn.Module, update_fn):
        with torch.no_grad():
            model_stdict = model.state_dict()
            for ema_n, ema_v in self.module.state_dict().items():
                try:
                    model_v = model_stdict[ema_n]
                except KeyError as e:
                    raise RuntimeError(
                        f"Paremeter {ema_n} not found when updating ema."
                    ) from e

                if self.device is not None:
                    model_v = model_v.to(device=self.device)

                ema_v.copy_(update_fn(ema_v, model_v))

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)

    def train(self, *_):
        return self
