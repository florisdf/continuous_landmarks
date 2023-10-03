from pathlib import Path
import sys

import torch
from tqdm import tqdm
import wandb

from .training_steps import TrainingSteps
from ..dataset.transforms import Compose, Normalize, \
    AbsToRelLdmks, RelToAbsLdmks


class TrainingLoop:
    def __init__(
        self,
        training_steps: TrainingSteps,
        optimizer,
        lr_scheduler,
        device,
        num_epochs,
        dl_train,
        dl_val_list,
        val_every,
        save_ckpts,
        best_metric,
        best_metric_ds,
        higher_is_better,
        ckpts_path,
    ):
        self.training_steps = training_steps
        self.model = self.training_steps.model.to(device)

        self.num_epochs = num_epochs
        self.dl_train = dl_train
        self.dl_val_list = dl_val_list

        self.val_every = val_every

        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_idx = 0
        self.train_batch_idx = -1

        self.save_ckpts = save_ckpts
        self.ckpts_path = Path(ckpts_path)

        self.best_metric_name = best_metric
        self.best_metric_value = None
        self.best_metric_ds = best_metric_ds
        self.higher_is_better = higher_is_better

    def run(self):
        self.minmax_metrics = {}

        for self.epoch_idx in tqdm(range(self.num_epochs), leave=True):
            self.training_steps.on_before_training_epoch()

            for (self.train_batch_idx, batch) in enumerate(
                tqdm(self.dl_train, leave=False),
                start=self.train_batch_idx + 1
            ):
                self._inner_loop(batch)

            log_dict = self.training_steps.on_after_training_epoch()
            log(log_dict, epoch_idx=self.epoch_idx, section='Train')

        # Create epoch checkpoint
        if self.save_ckpts:
            self.create_checkpoints(
                is_last=False,
                is_best=False,
                is_epoch=True,
                epoch_idx=self.epoch_idx,
            )

    def _inner_loop(self, train_batch):
        self.model.train()

        batch = tuple(x.to(self.device) for x in train_batch)
        loss, log_dict = self.training_steps.on_training_step(batch)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        log(log_dict, epoch_idx=self.epoch_idx,
            batch_idx=self.train_batch_idx,
            section='Train')
        log(dict(LR=self.lr_scheduler.get_last_lr()[0]),
            epoch_idx=self.epoch_idx,
            batch_idx=self.train_batch_idx)

        if self.train_batch_idx % self.val_every == 0:
            self.run_validation_epochs()

        if torch.isnan(loss):
            sys.exit('Loss is NaN. Exiting...')

    def run_validation_epochs(self):
        self.model.eval()

        log_dicts = {}
        for dl_val in self.dl_val_list:
            self.training_steps.on_before_validation_epoch()
            ds = dl_val.dataset
            inv_tfm = get_inv_tfm(ds.transform)
            for batch_idx, batch in tqdm(
                enumerate(dl_val),
                leave=False,
                total=len(dl_val),
            ):
                batch = tuple(x.to(self.device) for x in batch)
                with torch.no_grad():
                    self.training_steps.on_validation_step(
                        batch, batch_idx, inv_tfm
                    )

            d = self.training_steps.on_after_validation_epoch()
            ds_name = ds.__class__.__name__
            log(d, epoch_idx=self.train_batch_idx,
                section=f'Val{ds_name}')
            log_dicts[ds_name] = d

        # Create checkpoints
        if self.save_ckpts:
            self.create_checkpoints(
                is_last=True,
                is_best=self.metric_improved(log_dicts[self.best_metric_ds]),
                is_epoch=False,
            )

    def create_checkpoints(self, is_last, is_best, is_epoch,
                           epoch_idx=None):
        file_prefix = f"{wandb.run.id}_"
        file_suffix = '.pth'

        ckpt_dict = self.model.state_dict()

        if not self.ckpts_path.exists():
            self.ckpts_path.mkdir(parents=True)

        if is_best:
            torch.save(
                ckpt_dict,
                self.ckpts_path / f'{file_prefix}best{file_suffix}'
            )
        if is_last:
            torch.save(
                ckpt_dict,
                self.ckpts_path / f'{file_prefix}last{file_suffix}'
            )
        if is_epoch:
            assert epoch_idx is not None
            torch.save(
                ckpt_dict,
                self.ckpts_path / f'{file_prefix}ep{epoch_idx}{file_suffix}'
            )

    def metric_improved(self, val_log_dict):
        v = val_log_dict[self.best_metric_name]

        if self.best_metric_value is None:
            self.best_metric_value = v
            return True
        elif self.higher_is_better and v >= self.best_metric_value:
            self.best_metric_value = v
            return True
        elif not self.higher_is_better and v <= self.best_metric_value:
            self.best_metric_value = v
            return True
        else:
            return False


def log(log_dict, epoch_idx, batch_idx=None, section=None):
    def get_key(k):
        if section is None:
            return k
        else:
            return f'{section}/{k}'

    def get_value(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        else:
            return v

    for k, v in log_dict.items():
        wandb_dict = {get_key(k): get_value(v),
                      "epoch": epoch_idx}
        if batch_idx is not None:
            wandb_dict['batch_idx'] = batch_idx
        wandb.log(wandb_dict)


def get_inv_tfm(compose_tfm):
    norm_obj = compose_tfm.transforms[-1]
    assert isinstance(norm_obj, Normalize)
    inv_std = [1/s for s in norm_obj.std]
    inv_mean = [-m/s for m, s in zip(norm_obj.mean, norm_obj.std)]
    inv_norm = Normalize(inv_mean, inv_std)

    assert any(isinstance(t, AbsToRelLdmks) for t in compose_tfm.transforms)
    rel2abs = RelToAbsLdmks()

    return Compose([
        inv_norm,
        rel2abs
    ])
