import os
import argparse
import time
import shutil
from pytorch_lightning import callbacks

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from data_loader import get_segmentation_dataset
# from models.fast_scnn import get_fast_scnn
from models.fast_scnn_lightning import get_fast_scnn

from utils.metric import SegmentationMetric
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', type=str, default='val',
                        help='dataset validation split (default: val)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for training (default: 12)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=True,
                        help='skip validation during training')
    # the parser
    args = parser.parse_args()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True
    # args.device = device
    # print(args)
    # args_dict = vars(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)


        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(args.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))



        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        cur_iters = 0
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()

            for i, (images, targets) in enumerate(self.train_loader):
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                if cur_iters % 10 == 0:
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(self.train_loader),
                        time.time() - start_time, cur_lr, loss.item()))

            if self.args.no_val:
                # save every epoch
                save_checkpoint(self.model, self.args, is_best=False)
            else:
                self.validation(epoch)

        save_checkpoint(self.model, self.args, is_best=False)

    # def validation(self, epoch):
    #     is_best = False
    #     self.metric.reset()
    #     self.model.eval()
    #     for i, (image, target) in enumerate(self.val_loader):
    #         image = image.to(self.args.device)

    #         outputs = self.model(image)
    #         pred = torch.argmax(outputs[0], 1)
    #         pred = pred.cpu().data.numpy()
    #         self.metric.update(pred, target.numpy())
    #         pixAcc, mIoU = self.metric.get()
    #         print('Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (
    #             epoch, i + 1, pixAcc * 100, mIoU * 100))

    #     new_pred = (pixAcc + mIoU) / 2
    #     if new_pred > self.best_pred:
    #         is_best = True
    #         self.best_pred = new_pred
    #     save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.model, args.dataset)
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    


    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode="min")
    model_check_callback = ModelCheckpoint(dirpath=os.path.join(os.path.expanduser(args.save_folder), f"{args.model}_{args.dataset}.pth"),
                                            monitor='score', verbose=True, mode="max")

    pl_trainer = pl.Trainer(callbacks=[early_stop_callback], gpus=[0])
    # create network
    model = get_fast_scnn(namespace=args)
    pl_trainer.fit(model)




    # trainer = Trainer(args)
    # if args.eval:
    #     print('Evaluation model: ', args.resume)
    #     trainer.validation(args.start_epoch)
    # else:
    #     print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
    #     trainer.train()
