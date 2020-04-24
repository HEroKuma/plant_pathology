import pandas as pd
import cv2
import numpy as np
import os
import argparse
import tqdm
from model import *
from DataLoader import *

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='1')
    opt = parser.parse_args()
    print(opt)

    return opt

def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    cuda = True if torch.cuda.is_available() else False

    model = SimpleCNN().cuda()

    transforms_ = [transforms.Resize((512, 512), Image.BICUBIC),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(Data('./', transforms_), batch_size=opt.batch, shuffle=True, num_workers=opt.n_cpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    with tqdm.tqdm(total=opt.epoch, miniters=1, mininterval=0) as progress:
        for epoch in range(opt.n_epochs):
            for i, batch in enumerate(dataloader):
                images = Variable(batch['X']).cuda()
                label = Variable(batch['Y']).cuda()

                optimizer.zero_grad()
                outputs = model(images)
                _, correct_label = torch.max(outputs, 1)
                correct_num = (correct_label == label).sum()
                acc = correct_num.item()/label.size(0)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                if (i+1)%5 == 0:
                    progress.set_description("epoch: {epoch}, Iter {iter:.3f}, Loss: {loss}, acc: {acc}"\
                                             .format(epoch=epoch, iter=(i+1)/len(dataloader), loss=loss, acc=acc))
                    # print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %\
                    #       (epoch+1, 20, i+1, len(dataloader), loss.item(), acc))
            print('')

if __name__ == '__main__':
    opt = parse()
    train(opt)