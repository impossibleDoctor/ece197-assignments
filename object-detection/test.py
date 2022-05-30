import torch
import torchvision

import os

from download_utils import download_dataset, download_pretrained_model
from drinks_utils import get_drinks
import utils
from engine import evaluate
import presets

def get_transform(args):
    if args.weights:
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

    parser.add_argument("--model",                      default="fasterrcnn_mobilenet_v3_large_fpn", type=str, help="model name")
    parser.add_argument("--device",                     default="cuda",                 type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--data-augmentation",          default="hflip",                type=str, help="data augmentation policy (default: hflip)" )
    parser.add_argument("-j", "--workers",              default=4,                      type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--print-freq",                 default=50,                     type=int, help="print frequency")
    parser.add_argument("--output-dir",                 default="checkpoints",          type=str, help="path to save outputs")
    parser.add_argument("--sync-bn",                    action="store_true",            dest="sync_bn", help="Use sync batch norm", )
    parser.add_argument("--use-pretrained",             action="store_true",            dest="use_pretrained", help="Use pretrained model", )

    # distributed training parameters
    parser.add_argument("--world-size",                 default=1,                      type=int, help="number of distributed processes")
    parser.add_argument("--weights",                    default=None,                   type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp",                                                        action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


def main(args):
    if args.output_dir: utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)

    print(args)

    # LOAD DATASET
    download_dataset()
    dataset_test = get_drinks("drinks", "test", get_transform(args))

    device = torch.device("cuda")

    # CREATING DATA LOADERS
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    # CREATE MODEL
    num_classes = 4
    model = torchvision.models.detection.__dict__[args.model]( num_classes=num_classes, pretrained=False )
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # LOAD MODEL
    if args.use_pretrained:
        download_pretrained_model(args.model)
        pth_file_name = "drinks_{}.pth".format(args.model)
        model_path = os.path.join(args.output_dir, pth_file_name)
    else:
        model_path = os.path.join(args.output_dir, "checkpoint.pth")
        if not os.path.exists(model_path):
            print("Trained pth file not found. Do some training first.")
            print("Will use the default pretrained model.")

            model_name = "fasterrcnn_mobilenet_v3_large_fpn"
            download_pretrained_model(model_name)
            pth_file_name = "drinks_{}.pth".format(model_name)
            model_path = os.path.join(args.output_dir, pth_file_name)


    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    # EVALUATE MODEL
    evaluate(model, data_loader_test, device)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)