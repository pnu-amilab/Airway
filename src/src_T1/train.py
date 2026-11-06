import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--name_dset', type=str, default='Preset1')
parser.add_argument('--name_model', type=str, default='ASTRA_Net_phase_1')
parser.add_argument('--name_ablation', type=str, default='')
parser.add_argument('--epoch_load', type=int, default=0)
parser.add_argument('--full_train', action='store_true')
args = parser.parse_args()
if args.epoch_load not in (0, -1): parser.error('--epoch_load accepts only 0 (new training) or -1 (resume)')

import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_NUM)

import time
import numpy as np
import logging
import datetime
import warnings
import torch
warnings.filterwarnings(action='ignore')

path_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(path_src)
if path_src not in sys.path: sys.path.insert(0, path_src)

from src_T1.runners import get_runner
from src_T1.utils.utils_print import print_log
from src_T1.options.hyper_parameters import HP, load_local_HP

print(torch.cuda.is_available(), ': ', torch.cuda.get_device_name(0), args.GPU_NUM)
hp = HP(name_dset=args.name_dset, name_model=args.name_model, name_ablation=args.name_ablation, epoch_load=args.epoch_load, full_train=args.full_train)
load_local_HP(hp)
torch.save(hp, f'{hp.path_res}/options.pt')

logging.basicConfig(
    handlers=[
        logging.FileHandler(f'{hp.path_logs}/train.log', mode='a' if args.epoch_load == -1 else 'w'),
        logging.FileHandler(f'{hp.path_res}/train.log', mode='a' if args.epoch_load == -1 else 'w'),
    ],
    level=logging.INFO,
    format='%(message)s'
)
print_log(f'')
print_log(f'{torch.cuda.is_available()}: {torch.cuda.get_device_name(0)} {args.GPU_NUM}')
print_log(f'cwd: {os.getcwd()}')
print_log(f'--'*40)
for i in hp.__dict__: print_log(f'{i}: {hp.__dict__[i]}')

runner = get_runner(hp)
model = runner.build_model()
data_loader = runner.build_dataloader()
optimizer = runner.build_optimizer(model)
scheduler = runner.build_scheduler(optimizer, total_steps=hp.epochs*len(data_loader[0]))
scaler = runner.build_scaler()

history = []
epoch_s = 0
epoch_e = hp.epochs
if hp.epoch_load == -1:
    last_point = torch.load(f'{hp.path_model}/last_point.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(last_point['model'])
    runner.load_optimizer_state_dict(optimizer, last_point['optimizer'])
    runner.load_scheduler_state_dict(scheduler, last_point['scheduler'])
    scaler.load_state_dict(last_point['scaler'])
    history = last_point['history']
    epoch_s = last_point['epoch']+1
    print_log(f'※ Continual Learning from {epoch_s} Epoch ※')

print_log(f'--'*40)
print_log(f'{"Model":<20}: {hp.name}')
print_log(f'Loss: {runner.loss_keys} |* Rate: {runner.rate_keys}')
list_t = [0]
t0 = time.time()

for epoch in range(epoch_s, epoch_e):
    epoch_log = {'epoch':epoch}
    runner.set_epoch(epoch, epoch_e)
    train_log = runner.run_epoch(phase='Train', loader=data_loader[0], model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
    valid_log = runner.run_epoch(phase='Valid', loader=data_loader[1], model=model, optimizer=optimizer, scaler=scaler, scheduler=None)
    if scheduler is not None: scheduler.step()
    epoch_log['Train'] = train_log
    epoch_log['Valid'] = valid_log
    history.append(epoch_log)
    torch.save(history, f'{hp.path_model}/history.pt')

    if ((epoch+1)%hp.save_cycle == 0) or (epoch+1 == hp.epochs):
        torch.save(model.state_dict(), f'{hp.path_model}/model_{epoch+1}.pt')
        last_point = {'epoch':epoch, 'history':history, 'model':model.state_dict(), 'optimizer':runner.get_optimizer_state_dict(optimizer),
                      'scheduler':runner.get_scheduler_state_dict(scheduler), 'scaler':scaler.state_dict(), 'epochs':hp.epochs,
                      'len_train_loader':len(data_loader[0]), 'total_steps':hp.epochs*len(data_loader[0]), 'optimizer_lr':hp.optimizer_lr}
        torch.save(last_point, f'{hp.path_model}/last_point.pt')

    t1 = time.time()
    list_t.append(t1-t0)
    list_dt = [list_t[t+1]-list_t[t] for t in range(len(list_t)-1)]
    prev_t = np.sum(list_dt)
    next_t = np.mean(list_dt[-10:])*(epoch_e-epoch-1)
    total_t = prev_t+next_t
    tr_loss = epoch_log['Train']['loss']
    va_loss = epoch_log['Valid']['loss']
    tr_rate = epoch_log['Train']['rate']
    va_rate = epoch_log['Valid']['rate']

    print_log(f'{epoch+1:4.0f}/{hp.epochs:4.0f}', end='')
    print_log(f'|{list_dt[-1]/60:5.2f}:{list_t[-1]/60:7.2f}/{total_t/60:7.2f} m', end=' | ')
    for k in runner.loss_keys: print_log(f'{tr_loss[k]:6.4f}/{va_loss[k]:6.4f}', end=' | ')
    print_log('*', end=' ')
    for k in runner.rate_keys: print_log(f'{tr_rate[k]:6.4f}/{va_rate[k]:6.4f}', end=' | ')
    print_log(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end='')
    print_log()
    if (epoch+1)%20 == 0: print_log(f'{"Model":<20}: {hp.name}')

print_log(f'{"Model":<20}: {hp.name} [{torch.cuda.get_device_name(0)} {args.GPU_NUM}]')
