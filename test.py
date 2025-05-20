import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn import metrics
from torch.autograd import Variable

from datasets.factory import create_data_transforms
from datasets.ff_all import FaceForensics
from model.CUTA import CUTA
from utils.utils import *

torch.autograd.set_detect_anomaly(True)


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default='./config/FF++.yml', type=str, help="Path to option YMAL file.")
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--device', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--mixup', action="store_false",
                        help='using mixup augmentation.')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')  # 16
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')

    # swin argument
    parser.add_argument('--cfg', type=str, default="configs/swin_base_patch4_window12_384.yaml", metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
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

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    opt = yaml.safe_load(open(args.opt, 'r'))
    seed = opt["test"]["manual_seed"]

    if seed is not None:
        torch.cuda.manual_seed_all(seed)

    main_worker(args, opt)


def main_worker(args, opt):
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.gpu)

    print(f"Creating model: {opt['model']['baseline']}")
    model = CUTA(split='test')
    model.to(args.device)

    state_dict = torch.load(opt['test']['ckt_path'], map_location='cpu')
    model.load_state_dict(cleanup_state_dict(state_dict['state_dict']), strict=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Data loading code
    all_transform = create_data_transforms(opt)
    test_data = FaceForensics(opt, split='test', transforms=all_transform)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=opt['datasets']['test']['batch_size'], shuffle=False,
        num_workers=opt['datasets']['n_workers'], sampler=None, drop_last=False)

    test_acc, test_loss, test_auc = validate(test_loader, model, criterion, args, opt)


@torch.no_grad()
def validate(val_loader, model, criterion, args, opt):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(0))

    end = time.time()
    y_trues = []
    y_preds = []
    for i, (images, labels, _) in enumerate(val_loader):
        data_time.update(time.time() - end)

        images = images.to(args.device)
        labels = labels.to(args.device)

        # forward
        preds = model(images)

        # measure accuracy and record loss
        loss = criterion(preds, labels)  # preds[0]
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
            print(labels)
            progress.display(i)

    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    print(f' * Acc@1 {acc.avg:.3f}, auc {auc:.3f}')

    return acc.avg, losses.avg, auc


if __name__ == '__main__':
    main()
