import torch
import datetime
import time
import cv2
from glob import glob
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import argparse
import os
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
# from modules.dataloader.dataset import QioTrain, QioTest, QioVal
# from modules.arch.vgg import VGGNet
# from modules.arch.unet import UNet
# from modules.arch.fcn import FCN8s
# from modules.preprocessing.utils import save_inference_samples, get_image_paths, iou, pixel_acc
from preprocessing.dataloader.dataset import QioTrain, QioTest, QioVal, CustomTransform
from preprocessing.arch.vgg import VGGNet
from preprocessing.arch.unet import UNet
from preprocessing.arch.fcn import FCN8s
from preprocessing.modules.utils import save_inference_samples, get_image_paths, iou, pixel_acc
from torch.utils.data.sampler import SubsetRandomSampler

np.random.seed(1234)
torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True,
                    help='output directory for test inference')
parser.add_argument('--root_dir', type=str, required=True,
                    help='root directory for the dataset')
parser.add_argument('--model', type=str, default='vgg19',
                    help='model architecture to be used for FCN')
parser.add_argument('--use_model', type=str,
                    help='path to a saved model')
parser.add_argument('--epochs', type=int, default=85,
                    help='num of training epochs')
parser.add_argument('--n_class', type=int, default=2,
                    help='number of label classes')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay for L2 penalty')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='multiplicative factor of learning rate decay')
parser.add_argument('--step_size', type=int, default=20,
                    help='decay LR by a factor of gamma every step_size epochs')
parser.add_argument('--validate', type=int, default=1,
                    help='do inference on validation images (1) or test images (0)')
parser.add_argument('--acc_out', type=str, default="performance",
                    help='path to the directory where to save model performances')
parser.add_argument('--ignore_ids', type=str, default="general_outlier_imgs.csv",
                    help='path to the file with outlier image ids')
args = parser.parse_args()

# checking if a GPU is available (the available device would be used to train the model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# instantiating the model to use
vgg_model = VGGNet(model=args.model, requires_grad=True)
model = FCN8s(pretrained_net=vgg_model, n_class=args.n_class)
model = model.to(device)

# uncomment the lie below to use UNET instead of FCN
# model = UNet(3, args.n_class)

# uncomment the line below to use SGD instead of Adam optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
#                             momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                            lr=args.lr, weight_decay=args.weight_decay)

# Below is the code for 2 different types of schedulers to modify learning rate when using SGD
# Adam changes the learning rate on its own.

# scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
criterion = nn.BCELoss()

# qio_train = QioTrain(rootdir=args.root_dir, transform=transforms.Compose([
#                                                     CustomTransform(256)]))
qio_train = QioTrain(rootdir=args.root_dir)
df = pd.read_csv(args.ignore_ids, index_col=0)
ignore_ids = df["file_name"].tolist()

qio_val = QioTrain(rootdir=args.root_dir, transform=None, is_val=True,
                    ignore_ids=ignore_ids)
qio_test = QioTest(rootdir=args.root_dir)
# qio_test = QioVal(rootdir=args.root_dir)

# dataset = DataLoader(qio_train, batch_size=args.batch_size)
testloader = DataLoader(qio_test, batch_size=1)
# trainloader = DataLoader(qio_train, batch_size=args.batch_size, shuffle=True)
# valloader = DataLoader(qio_val, batch_size=1)

validation_split = 0.2
shuffle_dataset = True
random_seed = 425

dataset_size = qio_train.__len__()
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(qio_train, batch_size=args.batch_size,
                            sampler=train_sampler)
validation_loader = DataLoader(qio_val, batch_size=args.batch_size,
                            sampler=valid_sampler)

perf = pd.DataFrame(columns=['timestamp', 'epoch', 'pixel_acc', 'mean_iou', 'ious'])
# data_loaders = {"train": train_loader, "val": validation_loader}

def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger

def train(perf):
    for epoch in range(args.epochs):
        model.train()
        # uncomment the line below if using a scheduler with SGD
        # scheduler.step()
        ts = time.time()
        running_loss = 0.0
        for iter, batch in enumerate(train_loader):
            images = batch['image']
            images = images.float()
            labels = batch['label']
            labels = labels.float()
            images = Variable(images.to(device))
            labels = Variable(labels.to(device), requires_grad=False)
            optimizer.zero_grad()
            output = model(images)
            output = torch.sigmoid(output)
            loss = criterion(output, labels)
            loss.backward()
            # update the weights
            optimizer.step()

            running_loss += loss.item()
        print('epoch: %d/%d, train_loss: %.4f' %
              (epoch + 1, args.epochs,
              running_loss / len(train_loader)))
        if args.validate:
            perf = val(epoch, perf)
        print('time_elapsed: %.4f' % (time.time() - ts))
    torch.save(model.state_dict(), os.path.join(args.root_dir, "saved_model.pth"))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    perf.to_csv(timestr + ".csv")

    # uncomment and use this code to save the current model state with the load_checkpoint method
    # state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
    #          'optimizer': optimizer.state_dict(), 'losslogger': losslogger}
    # torch.save(state, filename)

def val(epoch, perf):
    # eval mode for validation phase
    model.eval()
    total_ious = []
    pixel_accs = []
    running_loss = 0.0
    for iter, batch in enumerate(validation_loader):
        images = batch['image']
        images = images.float()
        labels = batch['label']
        labels = labels.float()
        images = Variable(images.to(device))
        labels = Variable(labels.to(device), requires_grad=False)

        output = model(images)
        output = torch.sigmoid(output)
        loss = criterion(output, labels)
        running_loss += loss.item()

        output = output.detach().cpu().numpy()
        # converting output and label shape to desired shape for computation
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N, h, w)
        labels = labels.cpu().numpy()
        labels = np.argmax(labels, axis=1)
        target = labels.reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t, args.n_class))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch: {}/{}, val_loss: {}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch+1, args.epochs, \
                running_loss/len(validation_loader), pixel_accs, np.nanmean(ious), ious))
    perf = perf.append({'timestamp': datetime.datetime.now(), 'epoch': epoch+1,
                'pixel_acc': pixel_accs, 'mean_iou': np.nanmean(ious), 'ious': ious},
                ignore_index=True)
    return perf

if __name__ == "__main__":
    if args.use_model:
        vgg_model = VGGNet(model=args.model, requires_grad=True)
        model = FCN8s(pretrained_net=vgg_model, n_class=args.n_class)
        # model = UNet(3, args.n_class)
        model.load_state_dict(torch.load(args.use_model))
        print("Using saved model..")
    else:
        print("Training model..")
        if args.validate:
            perf = val(-1, perf)
        train(perf)
    print("Completed training!")
    print("Starting inference...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.acc_out):
        os.makedirs(args.acc_out)
    save_inference_samples(args.n_class, args.output_dir, testloader,
                            model, os.path.join(args.root_dir, "testing"))

    print("Inference completed!")
