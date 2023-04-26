
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import model as m
from torch.utils.data import *
import torchvision
from dataloader import DataSet

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

best_prec1 = 0
parser = argparse.ArgumentParser()
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")


def main():
    global args, best_prec1
    
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    full_arch_name = args.arch
    args.store_name = '_'.join(['TDN_', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    if dist.get_rank() == 0:
        check_rootfolders()


    model = m(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=(args.tune_from and args.dataset in args.tune_from))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = DataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,]),
        dense_sample=args.dense_sample)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler,drop_last=True)  

    val_dataset = DataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,]),
        dense_sample=args.dense_sample)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, sampler=val_sampler, drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    model = DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)


    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))


    if args.evaluate:
        val_loader.sampler.set_epoch(args.start_epoch)
        prec1, prec5, val_loss = validate(val_loader, model, criterion)
        if dist.get_rank() == 0:
            is_best = prec1 > best_prec1
            best_prec1 = prec1
            save_epoch = args.start_epoch + 1
            save_checkpoint(
                {
                    'epoch': args.start_epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'prec1': prec1,
                    'best_prec1': best_prec1,
                }, save_epoch, is_best)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch=epoch,)
        if dist.get_rank() == 0:
            tf_writer.add_scalar('loss/train', train_loss, epoch)
            tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
            tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loader.sampler.set_epoch(epoch)
            prec1, prec5, val_loss = validate(val_loader, model, criterion)
            if dist.get_rank() == 0:
                tf_writer.add_scalar('loss/test', val_loss, epoch)
                tf_writer.add_scalar('acc/test_top1', prec1, epoch)
                tf_writer.add_scalar('acc/test_top5', prec5, epoch)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                tf_writer.flush()
                save_epoch = epoch + 1
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'prec1': prec1,
                        'best_prec1': best_prec1,
                    }, save_epoch, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger=None, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                             top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))  # TODO
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            output = model(input)

            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            loss = reduce_tensor(loss)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
        .format(top1=top1, top5=top5, loss=losses)))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/%d_epoch_ckpt.pth.tar' % (args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, best_filename)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


if __name__ == '__main__':
    main()






#Other helper functions

def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]

class AverageMeter(object):
#Tracks current and average value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt