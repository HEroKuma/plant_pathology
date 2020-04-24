import pandas as pd
import cv2
import numpy as np
import os
import csv
import argparse
import tqdm
from model import *
from DataLoader import *

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch

transforms_ = [transforms.Resize((512, 512), Image.BICUBIC),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='1')
    opt = parser.parse_args()
    print(opt)

    return opt


def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    cuda = True if torch.cuda.is_available() else False

    model = SimpleCNN().cuda()

    data = Data('./', transforms_)
    data_size = len(data)
    indices = list(range(data_size))
    split = int(np.floor(.2*data_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_set = SubsetRandomSampler(train_indices)
    val_set = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch, sampler=train_set, num_workers=opt.n_cpu)
    val_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch, sampler=val_set, num_workers=opt.n_cpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    with tqdm.tqdm(total=opt.epoch, miniters=1, mininterval=0) as progress:
        for epoch in range(opt.n_epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                images = Variable(batch['X']).cuda()
                correct_label = Variable(batch['Y']).cuda()

                optimizer.zero_grad()
                outputs = model(images)
                _, label = torch.max(outputs, 1)
                correct_num = (correct_label == label).sum()
                acc = correct_num.item()/correct_label.size(0)
                loss = criterion(outputs, correct_label)
                loss.backward()
                optimizer.step()

                if (i+1) % 5 == 0:
                    progress.set_description("epoch: {epoch}, Iter {iter:.1f}%, Loss: {loss:.4f}"\
                                             .format(epoch=epoch, iter=100*(i+1)/len(train_loader), loss=loss))
                    # print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %\
                    #       (epoch+1, 20, i+1, len(dataloader), loss.item(), acc))

            model.eval()
            correct = 0
            test_loss = 0
            val_min = float('inf')
            for j, batch in enumerate(val_loader):
                images = Variable(batch['X']).cuda()
                correct_label = Variable(batch['Y']).cuda()
                outputs = model(images)
                _, label = torch.max(outputs, 1)
                correct += (correct_label == label).sum()
                loss = criterion(outputs, correct_label)
                test_loss += (opt.batch*loss.item())
            print("\tTest Loss: {loss:.4f}, acc: {acc:.2f}%".format(loss=test_loss/len(val_loader)\
                                                                   , acc=100*correct/(opt.batch*len(val_loader))),)
            if test_loss < val_min:
                val_min = test_loss
                torch.save(model.state_dict(), 'SimpleCNN')
                print('\t==Model Saved==')

    predict(model)


def predict(model):
    testLoader = DataLoader(testData('./', transforms_), batch_size=1, num_workers=2)
    model.eval()

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id','healthy','multiple_diseases','rust','scab'])
        for i, batch in enumerate(testLoader):
            predict_array = [batch[1][0], 0, 0, 0, 0]
            images = Variable(batch[0]).cuda()
            outputs = model(images)
            _, label = torch.max(outputs, 1)
            predict_array[label.item()+1] = 1
            writer.writerow(predict_array)


if __name__ == '__main__':
    opt = parse()
    train(opt)