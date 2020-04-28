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
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch

transforms_1 = [transforms.Resize((512, 512), Image.BICUBIC),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transforms_2 = [transforms.Resize((256, 256), Image.BICUBIC),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transforms_3 = [
    transforms.CenterCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[.3, .2, .1])
]

test_trans = [
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[0.3, 0.2, 0.1])
]

transforms_ = transforms_3


class DenseCrossEntropy(nn.Module):
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        output = torch.zeros((len(y_true), 4)).cuda()
        for i in range(len(y_true)):
            output[i][int(y_true[i].item())] = 1
        y_true = output
        logprobs = torch.log_softmax(y_pred, dim=-1)
        # print(logprobs, y_true)
        loss = -y_true * logprobs
        loss = torch.sum(loss)

        return loss.mean()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='1')
    opt = parser.parse_args()
    print(opt)

    return opt


def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num

    # model = SimpleCNN2().cuda()
    model = PlantModel(True).cuda()
    # model = ResNet18().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.95)
    # criterion = nn.CrossEntropyLoss()
    criterion = DenseCrossEntropy()

    data = Data('./', transforms_)
    """
    data_size = len(data)
    indices = list(range(data_size))
    split = int(np.floor(.2*data_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    """
    all_data = pd.read_csv('./train.csv')
    d = all_data.iloc[:, 0].values
    skf = StratifiedKFold(5, shuffle=True, random_state=412)
    targets = all_data.iloc[:, [1, 2, 3, 4]].values
    train_y = targets[:, 0] + targets[:, 1]*2+targets[:, 2]*3
    val_min = float('inf')
    model_name = ''

    for i_fold, (t, v) in enumerate(skf.split(d, train_y)):
        train_index = SubsetRandomSampler(t)
        val_index = SubsetRandomSampler(v)
        train_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch, sampler=train_index, num_workers=opt.n_cpu)
        val_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch, sampler=val_index, num_workers=opt.n_cpu)
        with tqdm.tqdm(total=opt.epoch, miniters=1, mininterval=0) as progress:
            for epoch in range(opt.n_epochs):
                model.train()
                test_correct = 0
                for i, batch in enumerate(train_loader, 1):
                    images = Variable(batch['X']).cuda()
                    correct_label = Variable(batch['Y']).cuda()
                    optimizer.zero_grad()
                    outputs = model(images)
                    # print(outputs)
                    _, label = torch.max(outputs, 1)
                    correct_num = (correct_label == label).sum()
                    test_correct += correct_num.item()
                    acc = test_correct/(correct_label.size(0)*i)
                    loss = criterion(outputs, correct_label)
                    loss.backward()
                    optimizer.step()

                    if i % 4 == 0:
                        progress.set_description(
                            "fold: {fold}, epoch: {epoch}, Iter {iter:.1f}%, Loss: {loss:.4f}, acc: {acc:.2f}%, lr: {lr:.5f}"\
                            .format(fold=i_fold, epoch=epoch, iter=100*(i+1)/len(train_loader),\
                            loss=loss, acc=acc*100, lr=scheduler.get_lr()[0])
                        )

                model.eval()
                correct = 0
                test_loss = 0
                for j, batch in enumerate(val_loader):
                    images = Variable(batch['X']).cuda()
                    correct_label = Variable(batch['Y']).cuda()
                    with torch.no_grad():
                        outputs = model(images)
                        _, label = torch.max(outputs, 1)
                        correct += (correct_label == label).sum()
                        loss = criterion(outputs, correct_label)
                    test_loss += loss.item()
                print("\tTest Loss: {loss:.4f}, acc: {acc:.2f}% {corr}/{data_size}".format(loss=test_loss/len(val_loader)\
                                                                       , acc=100*correct/(opt.batch*len(val_loader)),\
                                                                        corr=correct, data_size=len(val_index))
                                                                        )
                if test_loss < val_min:
                    val_min = test_loss
                    # torch.save(model.state_dict(), 'Res18')
                    torch.save(model, 'Res18_epoch.pth'.format(fold=i_fold, epoch=epoch))
                    model_name = 'Res18_epoch.pth'.format(fold=i_fold, epoch=epoch)
                    print('\t==Model Saved with {fold}-{epoch}=='.format(fold=i_fold, epoch=epoch))
                scheduler.step()

    predict(model_name)


def predict(name):
    testLoader = DataLoader(testData('./', test_trans), batch_size=1, num_workers=2)
    model = torch.load(name).cuda()

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        for i, batch in enumerate(testLoader):
            predict_array = [batch[1][0], 0, 0, 0, 0]
            images = Variable(batch[0]).cuda()
            with torch.no_grad():
                outputs = model(images)
                _, label = torch.max(outputs, 1)
                predict_array[label.item()+1] = 1
            writer.writerow(predict_array)


if __name__ == '__main__':
    opt = parse()
    train(opt)
    # predict('Res18')