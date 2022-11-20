import sys
import os

import warnings

from torchvision.io import read_image
from PIL import Image
from torchvision import transforms, datasets
import torch, torchvision 
import matplotlib.pyplot as plt
from torchvision import models

import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
import pybboxes as pbx

import torch, torchvision 
from torchvision import models
from torchsummary import summary
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

from model import CANNet

from utils import save_checkpoint
import pybboxes as pbx

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Subset

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import pandas as pd
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

parser = argparse.ArgumentParser(description='PyTorch CANNet')




#### need custom dataloader
class CustomDataLoader(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.labels = os.listdir(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.images_path,
                                self.labels[idx].replace('.txt', '.tif'))
        image = cv2.imread(img_name)
        
        label_file = os.path.join(self.labels_path, self.labels[idx])
        boxs = pd.read_csv(label_file,
                           delimiter=' ',
                           names=['id', 'x_min', 'y_min', 'w', 'h'])
        _, heat = get_heatmap(image, boxs, (32,32))
            
        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = torch.from_numpy(image)
                      
        heat_tensor = torch.from_numpy(heat[:,:, np.newaxis])
        heat_tensor = heat_tensor.permute(2, 0, 1)
        
        return img_tensor.to(device).float(), heat_tensor.to(device).float()



def get_heatmap(image, boxs, heatmap_size=None):
    h, w, _ = image.shape
    heatmap = np.zeros((h, w))

    for i, row in boxs.iterrows():
        x1, y1, x2, y2 = pbx.convert_bbox((row.x_min, row.y_min, row.w, row.h), from_type="yolo", to_type="voc", image_size=(w,h))
        start_point = (x1, y1) 
        end_point = (x2, y2)
        image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 10)
        centr_x = int(x1+((x2-x1)/2))
        centr_y = int(y1+((y2-y1)/2))
        image = cv2.circle(image, (centr_x,centr_y), radius=int(0.01*w), color=(255, 0, 255), thickness=-1)
        heatmap = cv2.circle(heatmap, (centr_x,centr_y), radius=int(0.01*w), color=(255, 0, 255), thickness=-1)
    
    #heatmap = cv2.GaussianBlur(heatmap,(91,91),cv2.BORDER_REFLECT_101) 

    if heatmap_size:
        heatmap = cv2.resize(heatmap, heatmap_size, interpolation = cv2.INTER_AREA)
        
    return image, heatmap



preprocess = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

images_path = '../input/xview-tiled/images/train'
labels_path = '../input/xview-tiled/labels/train/'

train_indices, test_indices, _, _ = train_test_split(
    range(len(os.listdir(images_path))),
    torch.ones(len(os.listdir(images_path))),
    test_size=0.40,
    random_state=42
)
 
train_indices, val_indices, _, _ = train_test_split(
    train_indices,
    torch.ones(len(train_indices)),
    test_size=0.50,
    random_state=42
)

batch_size = 26
train_ds = CustomDataLoader(images_path, labels_path, preprocess)
sub_dataset_train =Subset(train_ds, train_indices)
dataloader_train = torch.utils.data.DataLoader(sub_dataset_train, batch_size=batch_size, shuffle=True)

#dataset_val = Subset(CustomDataLoader(val_path, val_labels, preprocess))
sub_dataset_val =Subset(train_ds, val_indices)
dataloader_val = torch.utils.data.DataLoader(sub_dataset_val, batch_size=1, shuffle=True)

# dataset_test = Subset(CustomDataLoader(test_path, test_labels, preprocess))
sub_dataset_test =Subset(train_ds, test_indices)
dataloader_test = torch.utils.data.DataLoader(sub_dataset_test, batch_size=batch_size, shuffle=True)



def main():

    global args,best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size    = 26
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 100
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 4    


    torch.cuda.manual_seed(args.seed)

    model = CANNet()

    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.decay)

    for epoch in range(args.start_epoch, args.epochs):
        train(dataloader_train, model, criterion, optimizer, epoch)
        prec1 = validate(dataloader_val, model, criterion)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader), args.lr))

    model.train()
    end = time.time()

    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)[:,0,:,:]

        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    print ('begin val')
    model.eval()

    mae = 0
    for i,(img, target) in enumerate(val_loader):
        h,w = img.shape[2:4]
        h_d = h/2
        w_d = w/2
        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

        mae += abs(pred_sum-target.sum())

    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

if __name__ == '__main__':
    main()