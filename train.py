import argparse
import json
import os
import shutil
import time
import warnings

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from sklearn import metrics
from torch.autograd import Variable
from torch.optim import lr_scheduler

from datasets.factory import create_data_transforms
from datasets.ff_all import FaceForensics
from model.CUTA import CUTA
from utils.utils import *

torch.autograd.set_detect_anomaly(True)


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()

    parser.add_argument("--opt", default='./config/FF++.yml', type=str, help="Path to option YMAL file.")
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--device', default=0, type=int,
                        help='GPU id to use.')

    parser.add_argument('--mixup', action="store_false",
                        help='using mixup augmentation.')

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(verbose=True)
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')

    # swin argument
    parser.add_argument('--cfg', type=str, default="configs/swin_base_patch4_window12_384.yaml", metavar="FILE",
                        help='path to config file', )
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()

    opt = yaml.safe_load(open(args.opt, 'r'))
    seed = opt["train"]["manual_seed"]

    if seed is not None:
        # random.seed(args.seed)
        if args.gpu is None:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    else:
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args, opt)


def main_worker(gpu, ngpus_per_node, args, opt):
    args.gpu = gpu

    config = wandb.config
    config = vars(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(args.gpu)

    elif args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(args.local_rank)

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)
    else:
        pass

    # create model
    print(f"Creating model: {opt['model']['baseline']}")
    model = CUTA(split='train')
    model.init_weights()
    model.to(args.device)

    if not opt['train']['resume'] == None:
        from_pretrained(model, opt)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)
        param_groups = optim_factory.add_weight_decay(model.module, opt['train']['weight_decay'])
    else:
        param_groups = optim_factory.add_weight_decay(model, opt['train']['weight_decay'])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.Adam(param_groups, lr=opt['train']['lr'], betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    cudnn.benchmark = True

    # Data loading code
    all_transform = create_data_transforms(opt)
    train_data = FaceForensics(opt, split='train', transforms=all_transform)
    val_data = FaceForensics(opt, split='val', transforms=all_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, rank=args.local_rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, rank=args.local_rank)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt['datasets']['train']['batch_size'], shuffle=(train_sampler is None),
        num_workers=opt['datasets']['n_workers'], sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=opt['datasets']['train']['batch_size'], shuffle=False,
        num_workers=opt['datasets']['n_workers'], sampler=val_sampler, drop_last=False)

    if (args.gpu is not None or args.local_rank == 0) and opt['train']['resume'] == None:
        save_path = opt['train']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        var_dict = vars(args)
        var_dict['optimizer'] = str(optimizer.__class__.__name__)
        var_dict['device'] = str(args.device)
        json_str = json.dumps(var_dict)
        with open(os.path.join(save_path, 'config.json'), 'w') as json_file:
            json_file.write(json_str)

    best = 0.0

    for epoch in range(opt['train']['start_epoch'], opt['train']['epoch']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_acc, train_loss, train_celoss = train(train_loader, model, criterion,  # , train_segloss
                                                    optimizer, epoch, args, opt)
        test_acc, test_loss, test_auc = validate(val_loader, model, criterion, epoch, args, opt)
        scheduler.step()

        is_best = (test_auc + test_acc) > best
        best = max((test_auc + test_acc), best)

        if args.gpu is not None or args.local_rank == 0:
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if args.gpu == None else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, epoch=epoch)

            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']

            log_info = {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                'train_celoss': train_celoss,
                "test_acc": test_acc,
                "test_loss": test_loss,
                'test_auc': test_auc,
                'learning_rate': cur_lr
            }
            file_path = './log_info.txt'
            with open(file_path, 'a') as file:
                for key, value in log_info.items():
                    file.write(f"{key}: {value}\n")


def train(train_loader, model, criterion, optimizer, epoch, args, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ce = AverageMeter('Loss_ce', ':.5f')
    losses = AverageMeter('Loss', ':.5f')
    acc = AverageMeter('Acc@1', ':6.3f')
    timestamp = time.strftime('%Y%m%d_%H%M%S ', time.localtime())
    progress = ProgressMeter(
        len(train_loader),
        [timestamp, batch_time, data_time, losses_ce, losses, acc],  # losses_seg,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, labels, masks) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device).float()
        masks[masks > 0] = 1.0

        # forward
        preds = model(images)
        loss_ce = criterion(preds, labels)

        # measure accuracy and record loss
        loss = loss_ce
        acc1 = accuracy(preds, labels, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        acc.update(acc1, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (args.gpu is not None or args.local_rank == 0) and i % 20 == 0:
            progress.display(i)

    return acc.avg, losses.avg, losses_ce.avg


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args, opt):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.3f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    y_trues = []
    y_preds = []
    for i, (images, labels, masks) in enumerate(val_loader):
        data_time.update(time.time() - end)

        images = images.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device).float()

        masks[masks > 0] = 1.0
        # forward
        preds = model(images)

        # measure accuracy and record loss
        loss = criterion(preds, labels)
        acc1 = accuracy(preds, labels, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        acc.update(acc1, images.size(0))

        y_trues.extend(labels.cpu().numpy())
        prob = 1 - torch.softmax(preds, dim=1)[:, 0].cpu().numpy()
        y_preds.extend(prob)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (args.gpu is not None or args.local_rank == 0) and i % 20 == 0:
            progress.display(i)

    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    print(f' * Acc@1 {acc.avg:.4f}, auc {auc:.4f}')

    return acc.avg, losses.avg, auc


def save_checkpoint(state, is_best, epoch, file='./checkpoints/'):
    if not os.path.exists(file):
        os.makedirs(file)
    filename = os.path.join(file, 'checkpoint-{:02d}.pth.tar'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(file, 'model_best.pth.tar'))


def from_pretrained(model, opt):
    state_dict = torch.load(opt['train']['resume'], map_location='cpu')
    model.load_state_dict(cleanup_state_dict(state_dict['state_dict']), strict=False)

    opt['train']['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
    opt['train']['start_epoch'] = state_dict['epoch']


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == '__main__':
    main()
