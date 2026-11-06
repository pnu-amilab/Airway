import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Runner_Base():

    def __init__(self, hp):
        self.hp = hp
        self.loss_keys = []
        self.rate_keys = []

    def set_epoch(self, epoch, epoch_e):
        pass

    def build_model(self):
        raise NotImplementedError

    def build_dataset(self):
        raise NotImplementedError

    def build_dataloader(self):
        train_set, valid_set = self.build_dataset()
        kwargs = {'batch_size':self.hp.batch_size, 'num_workers':self.hp.num_workers, 'pin_memory':torch.cuda.is_available()}
        if self.hp.num_workers > 0:
            kwargs['prefetch_factor'] = 2
            kwargs['persistent_workers'] = True
        train_loader = DataLoader(dataset=train_set, shuffle=True, **kwargs)
        valid_loader = DataLoader(dataset=valid_set, shuffle=False, **kwargs)

        return [train_loader, valid_loader]

    def get_batch(self, data):
        return data[0].to(self.hp.device), data[1].to(self.hp.device)

    def preprocess_batch(self, data):
        return self.get_batch(data)

    def forward(self, *args):
        raise NotImplementedError

    def forward_loss(self, *args):
        raise NotImplementedError

    def forward_rate(self, *args):
        raise NotImplementedError

    def run_step(self, phase, data, model, optimizer, scaler, scheduler):
        with torch.cuda.amp.autocast(enabled=self.hp.amp):
            res = self.forward(data, model)
            loss_dict, loss_final = self.forward_loss(res)

        if phase == 'Train':
            self._optimizer_zero_grad(optimizer)
            scaler.scale(loss_final).backward()
            self._scaler_unscale(scaler, optimizer)
            self._clip_gradients(model)
            self._scaler_step(scaler, optimizer)
            scaler.update()
            self._scheduler_step(scheduler)

        with torch.no_grad(): rate_dict = self.forward_rate(res)

        return loss_dict, rate_dict

    def run_epoch(self, phase, loader, model, optimizer, scaler, scheduler):
        if phase == 'Train':
            model.train()
            context_manager = torch.enable_grad()
        else:
            model.eval()
            context_manager = torch.no_grad()

        loss_sum, rate_sum = {}, {}
        n_batch = 0
        iter_loader = tqdm(loader, desc=phase, leave=False) if self.hp.run_tqdm else loader
        with context_manager:
            for data in iter_loader:
                n_batch += 1
                data = self.preprocess_batch(data)
                loss_dict, rate_dict = self.run_step(phase, data, model, optimizer, scaler, scheduler)
                for k, v in loss_dict.items(): loss_sum[k] = loss_sum.get(k, 0.0)+float(v.item())
                for k, v in rate_dict.items(): rate_sum[k] = rate_sum.get(k, 0.0)+float(v.item())

        loss_mean = {k:loss_sum.get(k, 0.0)/n_batch for k in self.loss_keys}
        rate_mean = {k:rate_sum.get(k, 0.0)/n_batch for k in self.rate_keys}

        return {'loss':loss_mean, 'rate':rate_mean}

    def build_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.hp.optimizer_lr, weight_decay=self.hp.optimizer_wd)

    def build_scheduler(self, optimizer, total_steps):
        warmup_steps = int(self.hp.scheduler_warmup_ratio*total_steps)

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                (step/max(1, warmup_steps)) if step < warmup_steps
                else 0.5*(1.0+np.cos(np.pi*((step-warmup_steps)/max(1, total_steps-warmup_steps))))))

    def build_scaler(self):
        return torch.cuda.amp.GradScaler(enabled=self.hp.amp)

    def _clip_gradients(self, model): nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip_grad_norm)
    def _optimizer_zero_grad(self, optimizer): optimizer.zero_grad()
    def _scaler_unscale(self, scaler, optimizer): scaler.unscale_(optimizer)
    def _scaler_step(self, scaler, optimizer): scaler.step(optimizer)
    def _scheduler_step(self, scheduler):
        if scheduler is not None: scheduler.step()
    def get_optimizer_state_dict(self, optimizer): return optimizer.state_dict()
    def load_optimizer_state_dict(self, optimizer, state_dict): optimizer.load_state_dict(state_dict)
    def get_scheduler_state_dict(self, scheduler): return scheduler.state_dict()
    def load_scheduler_state_dict(self, scheduler, state_dict): scheduler.load_state_dict(state_dict)
    def get_last_lr(self, optimizer): return optimizer.param_groups[0]['lr']
