import os
import argparse

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from LitTransformer import LitTransformer
from LitSPEECHCOMMANDS import LitSPEECHCOMMANDS

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',           default="dataset/",     type=str,       help='dataset and checkpoint save path')
    parser.add_argument('--batch-size',     default=64,             type=int,       help="batch size")
    parser.add_argument('--lr',             default=0.0003,         type=float,     help="learning rate")
    parser.add_argument('--max-epochs',     default=30,             type=int,       help="maximum number of epochs")
    parser.add_argument('--num-classes',    default=37,             type=int,       help="number of classes") 

    parser.add_argument('--accelerator',    default='gpu',          type=str,       metavar='N')
    parser.add_argument('--devices',        default=1,              type=int,       metavar='N')
    parser.add_argument('--num-workers',    default=4,              type=int,       metavar='N')
    parser.add_argument('--precision',      default=16)

    parser.add_argument('--patch-num',      default=14,             type=int,       help="number of patches" )
    parser.add_argument('--n-fft',          default=512,            type=int,       help="number of fft for melspectrogram" )
    parser.add_argument('--n-mels',         default=42,             type=int,       help="number of mel for melspectrogram" )
    parser.add_argument('--win-length',     default=None,           type=int,       help="win length for melspectrogram" )
    parser.add_argument('--hop-length',     default=164,            type=int,       metavar='hop lenght for melspectrogram')

    parser.add_argument('--depth',          default=12,             type=int,       help='depth')
    parser.add_argument('--embed-dim',      default=64,             type=int,       help='embedding dimension')
    parser.add_argument('--num-heads',      default=2,              type=int,       help='num_heads')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)

    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    
    # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    datamodule = LitSPEECHCOMMANDS(
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        path=args.path,
                        patch_num=args.patch_num,
                        n_fft=args.n_fft,
                        n_mels=args.n_mels,
                        win_length=args.win_length,
                        hop_length=args.hop_length,
                        class_dict=CLASS_TO_IDX
                    )
    datamodule.setup()

    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    print("Patch dim:", patch_dim)
    print("Sequence length:", seqlen)

    model = LitTransformer(
            num_classes=args.num_classes,
            lr=args.lr,
            epochs=args.max_epochs, 
            depth=args.depth,
            embed_dim=args.embed_dim,
            head=args.num_heads,
            patch_dim=patch_dim,
            seqlen=seqlen
        )

    cpkt_name = "transformer-kws-best-acc"
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.path, "checkpoints"),
        filename=cpkt_name,
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    trainer = Trainer(accelerator=args.accelerator,
                        devices=args.devices,
                        precision=args.precision,
                        max_epochs=args.max_epochs,
                        logger= None,
                        callbacks=[model_checkpoint] )

    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class
    
    # TRAINING
    trainer.fit(model, datamodule=datamodule)

    # EVALUATION
    trainer.test(model, datamodule=datamodule)

    # SAVING
    ckpt_file = os.path.join( args.path, "checkpoints", cpkt_name+".ckpt")
    model = model.load_from_checkpoint(ckpt_file)
    model.eval()
    script = model.to_torchscript()

    # save for use in production environment
    model_path = os.path.join(args.path, "checkpoints", cpkt_name+".pt")
    torch.jit.save(script, model_path)