#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import builtins
import cv2
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from datetime import datetime
from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_optimizer, get_model, adjust_learning_rate
from utils.evaluate_utils import save_results_to_disk, \
                                 eval_segmentation_supervised_offline
from utils.logger import Logger
from utils.ema import EMA
from utils.utils import AverageMeter, ProgressMeter, SemsegMeter, freeze_layers
from termcolor import colored


# Parser
parser = argparse.ArgumentParser(description='Finetuning for semantic segmentation.')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--model_path', default=None,
                    help='path to model for evaluation')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--crf-postprocess', action='store_true',
                    help='Apply CRF post-processing')
parser.add_argument('--crf-train', action='store_true',
                    help='Apply CRF post-processing to train set')

cv2.setNumThreads(1) # set threads to 0 or 1 to prevent opencv from overloading your system
def main():
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('using GPUs', ngpus_per_node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    p = create_config(args.config_env, args.config_exp)

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu == 0:
        sys.stdout = Logger(p['log_file'])

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Create model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)

    # Enable sync batchnorm when training on multiple GPUs
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    print(model)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(p['train_db_kwargs']['batch_size'] / ngpus_per_node)
            args.workers = int((p['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                    find_unused_parameters=True)

        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    print(colored('Get loss', 'blue'))
    if p['loss'] == 'CE': # use standard cross-entropy loss
        criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(args.gpu)
    elif p['loss'] == 'ohem': # use hard pixel mining loss
        from losses import CrossEntropyOHEM
        criterion = CrossEntropyOHEM(**p['loss_kwargs']).cuda(args.gpu)
    print(criterion)

    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model.parameters())
    print(optimizer)

    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        if args.gpu is None:
            checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(p['checkpoint'], map_location=loc)

        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_iou = checkpoint['best_iou']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        best_epoch = 0
        best_iou = 0

    cudnn.benchmark = True

    print(colored('Getting train dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms) 
    base_dataset = get_train_dataset(p, val_transforms) 
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # no reshape
    print(colored('Train samples %d' %(len(train_dataset)), 'yellow'))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p['val_db_kwargs']['batch_size'], 
                    num_workers=p['num_workers'], pin_memory=True, shuffle=False, drop_last=False)
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=p['val_db_kwargs']['batch_size'], 
                    num_workers=p['num_workers'], pin_memory=True, shuffle=False, drop_last=False)

    if 'use_ema' in p.keys() and p['use_ema']:
        ema = EMA(model, alpha=p['alpha_ema'])
    else:
        ema = None

    if args.model_path is not None:
        assert p['val_db_name'] == 'VOCSegmentation'
        print(colored('Evaluating model', 'blue'))
        model.load_state_dict(torch.load(args.model_path))
        # save eval set to disk (used to report results)
        save_results_to_disk(p, val_loader, model, crf_postprocess=args.crf_postprocess)
        eval_stats = eval_segmentation_supervised_offline(p, true_val_dataset, verbose=True)
        return

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))
        if args.distributed:
            train_sampler.set_epoch(epoch)

        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # train for one epoch
        print('Training ... ')
        start = datetime.now()
        train(p, train_loader, model, criterion, optimizer, epoch, args.gpu, ema=ema)
        print('Training took {}'.format(datetime.now()-start))

        # validate
        print('Validate ...')
        start = datetime.now()
        eval_val = validate(p, val_loader, model, args.gpu)
        print('Validation took {}'.format(datetime.now()-start))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_iou': best_iou}, 
                        p['checkpoint'])
        
            # only used for validation, not for training
            # we selected the average of the last models to obtain the numbers
            if eval_val['mIoU'] > best_iou:
                print('Found new best model: %.2f -> %.2f (mIoU)' %(100*best_iou, 100*eval_val['mIoU']))
                best_iou = eval_val['mIoU']
                best_epoch = epoch
                torch.save(model.state_dict(), p['best_model'])
        
            else:
                print('No new best model: %.2f -> %.2f (mIoU)' %(100*best_iou, 100*eval_val['mIoU']))
                print('Last best model was found in epoch %d' %(best_epoch))

    if p['val_db_name'] == 'VOCSegmentation':
        print(colored('Evaluating model at the end', 'blue'))
        model.load_state_dict(torch.load(p['best_model']))
        if args.crf_train:
            # save train set to disk 
            assert args.crf_postprocess
            save_results_to_disk(p, base_loader, model, crf_postprocess=args.crf_postprocess)
        else:
            # save eval set to disk (used to report results)
            save_results_to_disk(p, val_loader, model, crf_postprocess=args.crf_postprocess)
            eval_stats = eval_segmentation_supervised_offline(p, true_val_dataset, verbose=True)
    else:
        pass


def train(p, train_loader, model, criterion, optimizer, epoch, gpu, ema=None):
    losses = AverageMeter('Loss', ':.4e')
    semseg_meter = SemsegMeter(p['num_classes'], train_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    if p['freeze_layers']:
        model = freeze_layers(model, p['freeze_layers'], p['model_kwargs']['use_fcn'])

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(gpu, non_blocking=True)
        targets = batch['semseg'].cuda(gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)

        losses.update(loss.item())
        semseg_meter.update(torch.argmax(output, dim=1), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 100 == 0:
            progress.display(i)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results

@torch.no_grad()
def validate(p, val_loader, model, gpu):
    semseg_meter = SemsegMeter(p['num_classes'], val_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    model.eval()

    for i, batch in enumerate(val_loader):
        images = batch['image'].cuda(gpu, non_blocking=True)
        targets = batch['semseg'].cuda(gpu, non_blocking=True)
        output = model(images)
        semseg_meter.update(torch.argmax(output, dim=1), targets)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results


if __name__ == "__main__":
    main()
