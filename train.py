import torch
import torchvision

import utils
import presets
from drinks_utils import get_drinks
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate
from download_utils import download_dataset

import time
import os
import datetime


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    # FIXED ARGS (NOT CONFIGURABLE)

    # parser.add_argument("--data-path",                  default="drinks",     type=str, help="dataset path" )
    # parser.add_argument("--dataset",                    default="drinks",                 type=str, help="dataset name")
    # parser.add_argument("--model",                      default="fasterrcnn_resnet50_fpn ", type=str, help="model name")
    
    parser.add_argument("--device",                     default="cuda",                 type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size",           default=2,                      type=int, help="images per gpu, the total batch size is $NGPU x batch_size" )
    parser.add_argument("--epochs",                     default=26,                     type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers",              default=4,                      type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--opt",                        default="sgd",                  type=str, help="optimizer")
    parser.add_argument("--lr",                         default=0.02,                   type=float, help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu", )
    parser.add_argument("--momentum",                   default=0.9,                    type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay",       default=1e-4,                   type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay", )
    parser.add_argument("--norm-weight-decay",          default=None,                   type=float, help="weight decay for Normalization layers (default: None, same value as --wd)",)
    parser.add_argument("--lr-scheduler",               default="multisteplr",          type=str, help="name of lr scheduler (default: multisteplr)" )
    parser.add_argument( "--lr-step-size",              default=8,                      type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)" )
    parser.add_argument("--lr-steps",                   default=[16, 22],               nargs="+",type=int,help="decrease lr every step-size epochs (multisteplr scheduler only)", )
    parser.add_argument("--lr-gamma",                   default=0.1,                    type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)")
    parser.add_argument("--print-freq",                 default=50,                     type=int, help="print frequency")
    parser.add_argument("--output-dir",                 default="checkpoints",          type=str, help="path to save outputs")
    parser.add_argument("--resume",                     default="",                     type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch",                default=0,                      type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor",  default=3,                      type=int )
    parser.add_argument("--data-augmentation",          default="hflip",                type=str, help="data augmentation policy (default: hflip)" )
    parser.add_argument("--sync-bn",                    action="store_true",            dest="sync_bn", help="Use sync batch norm", )

    # distributed training parameters
    parser.add_argument("--world-size",                 default=1,                      type=int, help="number of distributed processes")
    parser.add_argument("--dist-url",                   default="env://",               type=str, help="url used to set up distributed training")
    parser.add_argument("--weights",                    default=None,                   type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone",           default=None,                   type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp",                                                        action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # OUTDATED ARGS
    "--rpn-score-thresh"
    "--trainable-backbone-layers"

    return parser

def main(args):
    if args.output_dir: utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # DATA LOADING -------------------------------------
    print("Loading dataset")
    download_dataset()

    num_classes = 4 # somehow 3 classes don't work
    dataset = get_drinks("drinks", "train", get_transform(True, args))
    dataset_test = get_drinks("drinks", "test", get_transform(False, args))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )


    # CREATE MODEL ----------------------
    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # START TRAINING ----------------------
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)