import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor,load_pretained
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning.utils import distributed as pml_dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('MetaFG training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path',default='./imagenet', type=str, help='path to dataset')
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
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    parser.add_argument('--num-workers', type=int, 
                        help="num of workers on dataloader ")
    
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        help='weight decay (default: 0.05 for adamw)')
    
    parser.add_argument('--min-lr', type=float,
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float,
                        help='warmup learning rate')
    parser.add_argument('--epochs', type=int,
                        help="epochs")
    parser.add_argument('--warmup-epochs', type=int,
                        help="epochs")
    
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--lr-scheduler-name', type=str,
                        help='lr scheduler name,cosin linear,step')
    
    parser.add_argument('--pretrain', type=str,
                        help='pretrain')
    
    parser.add_argument('--tensorboard', action='store_true', help='using tensorboard')
    
    
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class InfoNCE(torch.nn.Module):
    def __init__(self, batch_size=8, temperature=0.07, is_triplet=True):
        super().__init__()
        self.temperature = temperature
        self.is_triplet = is_triplet
        self.batch_size = batch_size
        self.criterion = LabelSmoothingCrossEntropy() #torch.nn.CrossEntropyLoss()

    def __call__(self, features, labels):
        features = normalize(features)

        labels = []
        for i in range(8):
            labels.extend(list(range(i*self.batch_size,(i+1)*self.batch_size)))
            labels.extend(list(range(i*self.batch_size,(i+1)*self.batch_size)))

        # print('labels', features.shape, len(labels))
        labels = torch.tensor(labels).to(features.device)
        # labels = torch.arange(features.shape[0]).to(features.device)
        similarity_matrix = torch.matmul(features, features.T)
        # print('la', labels.shape, similarity_matrix.shape, labels)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # print('load', labels.sum(dim=-1), labels.sum(), labels.shape)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        # print('labels', labels.shape)
        labels = labels[~mask].view(labels.shape[0], -1) # remove self
        # print('dfff', labels.shape, similarity_matrix.shape)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # print('llsldd', labels.sum(dim=-1))
        # print('dfssssff', similarity_matrix.shape, similarity_matrix[labels.bool()].shape)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        negatives_max, _ = torch.max(negatives, dim=-1)
        
        logits1 = logits / self.temperature #/ self.temperature

        acc = (logits1.max(1)[1] == labels).float().mean()
        loss = self.criterion(logits1, labels)
        if self.is_triplet:
            margin_loss = torch.mean(
                torch.max(
                    torch.zeros(negatives_max.size(0)).to(device=negatives_max.device), 0.3 + negatives_max - positives.squeeze()
                )
            ) 
            return margin_loss + loss
        return loss

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    if config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    if config.MODEL.PRETRAINED:
        load_pretained(config,model_without_ddp,logger)
        if config.EVAL_MODE:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            return

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        logger.info(f"**********normal test***********")
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # if config.DATA.ADD_META:
        #     logger.info(f"**********mask meta test***********")
        #     acc1, acc5, loss = validate(config, data_loader_val, model,mask_meta=True)
        #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    loss_fn = InfoNCE(batch_size=config.DATA.BATCH_SIZE)
    loss_fn = pml_dist.DistributedLossWrapper1(loss=loss_fn) #, device_ids=[args.rank])

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)      
        train_one_epoch_local_data(config, model, criterion, loss_fn, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        if epoch % 3 == 0:
            logger.info(f"**********normal test***********")
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        # if config.DATA.ADD_META:
        #     logger.info(f"**********mask meta test***********")
        #     acc1, acc5, loss = validate(config, data_loader_val, model,mask_meta=True)
        #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
#         data_loader_train.terminate()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch_local_data(config, model, criterion, loss_fn, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,tb_logger=None):
    model.train()
    if hasattr(model.module,'cur_epoch'):
        model.module.cur_epoch = epoch
        model.module.total_epoch = config.TRAIN.EPOCHS
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):

        if config.DATA.ADD_META:
            samples, targets, meta = data
            # print('idx', samples.shape, targets.shape, meta.shape)
            raw_samples, aug_samples = samples[:,:3,:,:], samples[:,3:,:,:],
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            meta = meta.cuda(non_blocking=True)
        else:
            samples, targets= data
            raw_samples, aug_samples = samples[:,:3,:,:], samples[:,3:,:,:],
            meta = None

        raw_samples = raw_samples.cuda(non_blocking=True)
        aug_samples = aug_samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        samples = torch.cat((raw_samples, aug_samples), dim=0)
        targets = torch.cat((targets, targets), dim=0)

        # print('samples', samples.shape, targets.shape)
        test_data = []
        for i, t in enumerate(targets):
            # print(i, t.shape, t)
            if t != -1:
                test_data.append(i)

        # print('sss', raw_samples.shape, aug_samples.shape, targets.shape)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)


        # print('sssssalhueo', samples.shape, targets.shape)
        if config.DATA.ADD_META:
            meta = torch.cat((meta, meta), dim=0)
            feats, outputs = model(samples, meta)
        else:
            feats, outputs = model(samples)

        # print('feats', feats.shape, outputs.shape)
        temp_outputs = []
        temp_targets = []

        for i in test_data:
            temp_outputs.append(outputs[i].unsqueeze(0))
            temp_targets.append(targets[i].unsqueeze(0))

        if len(temp_outputs) != 0:
            temp_outputs = torch.cat(temp_outputs, dim=0)
            temp_targets = torch.cat(temp_targets, dim=0)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if len(temp_outputs) != 0:
                loss = criterion(temp_outputs, temp_targets) + 0.001*loss_fn(feats, targets) #loss_fn(feats, targets) #criterion(temp_outputs, temp_targets) + 0.001*loss_fn(feats, targets)
            else:
                targets = targets + 1
                loss = 0.000001*criterion(outputs, targets)+loss_fn(feats, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            # if config.AMP_OPT_LEVEL != "O0":
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     if config.TRAIN.CLIP_GRAD:
            #         grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            #     else:
            #         grad_norm = get_grad_norm(amp.master_params(optimizer))
            # else:
            if True:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(temp_outputs, temp_targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        # break
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, mask_meta=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    snake_test_result = []

    end = time.time()
    for idx, data in enumerate(tqdm(data_loader)):
        if config.DATA.ADD_META:
            images,target,meta = data
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            if mask_meta:
                meta = torch.zeros_like(meta)
            meta = meta.cuda(non_blocking=True)
        else:
            images, target = data
            meta = None
        
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if config.DATA.ADD_META:
            feats, output = model(images,meta)
        else:
            feats, output = model(images)

        if config.DATA.DATASET in ['snakeclef2022test', 'snakeclef2022valid']:
            for idx_b in range(len(target)):
                snake_test_result.append((target[idx_b].cpu(), output[idx_b].cpu()))
            continue

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    if config.DATA.DATASET == 'snakeclef2022test':
        if config.DATA.ADD_META and mask_meta:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2022test_mask_meta.tc')
        else:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2022test.tc')
        torch.save(snake_test_result, output_file)
        print(len(snake_test_result), output_file)
    elif config.DATA.DATASET == 'snakeclef2022valid':
        if config.DATA.ADD_META and mask_meta:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2022valid_mask_meta.tc')
        else:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2022valid.tc')
        torch.save(snake_test_result, output_file)
        print(len(snake_test_result), output_file)

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    # if config.AMP_OPT_LEVEL != "O0":
    #     assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}",local_rank=config.LOCAL_RANK)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
