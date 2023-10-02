import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil

from ..utils.draw_points import draw_points


class TrainingSteps:
    def __init__(
        self,
        pos_encoder,
        feat_extractor,
        lm_predictor,
        max_logged_ims=20,
    ):
        self.pos_encoder = pos_encoder
        self.feat_extractor = feat_extractor
        self.lm_predictor = lm_predictor
        self.max_logged_ims = max_logged_ims

        self.val_losses = []
        self.val_ims = []

    def on_before_training_epoch(self):
        pass

    def on_training_step(self, batch):
        lm_pred, loss = self._get_pred_and_loss(batch)
        log_dict = {
            'GaussianNLL': loss,
        }

        return loss, log_dict

    def _get_pred_and_loss(self, batch):
        img_batch, lm_true, canon_batch = batch
        B, N, _ = canon_batch.shape
        query_sequence = self.pos_encoder(
            canon_batch.flatten(end_dim=1)
        ).unflatten(0, (B, N))
        feature = self.feat_extractor(img_batch)
        *lm_pred, var_pred = self.lm_predictor(query_sequence, feature)

        loss = F.gaussian_nll_loss(lm_pred, lm_true, var_pred)

        return lm_pred, loss

    def on_after_training_epoch(self):
        return {}

    def on_before_validation_epoch(self):
        pass

    def on_validation_step(self, batch, batch_idx, inv_norm):
        lm_pred, loss = self._get_pred_and_loss(batch)
        self.val_losses.append(loss)

        if len(self.val_ims) < self.max_logged_ims:
            # Log images with LM preds
            img_batch, *_ = batch
            img = img_batch[0]
            img_batch = inv_norm(img)
            im = to_pil(img_batch.cpu())
            im = draw_points(im, lm_pred)
            self.val_ims.append(im)

    def on_after_validation_epoch(self):
        log_dict = {
            'GaussianNLL': torch.tensor(self.val_losses).mean()
        }

        self.val_losses.clear()
        self.val_ims.clear()

        return log_dict


def compute_and_get_log_dict(metric, suffix=''):
    result_dict = metric.compute()
    metric_name = metric.__class__.__name__

    log_dict = {}

    for k, v in result_dict.items():
        name = f'{metric_name}{suffix}/{k}'
        log_dict[name] = v
    return log_dict
