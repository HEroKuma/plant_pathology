import pandas as pd
import cv2
import numpy as np
import os
import torch
import csv
import argparse
import tqdm
import logging
from models import WSDAN
from DataLoader import *
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
from efficientnet_pytorch import EfficientNet


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--beta', type=float, default=5e-2)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--num_attentions', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log', type=str, default='efficientnet-b7.log')
    parser.add_argument('--net', type=str, default='efficientnet-b7')
    # parser.add_argument('--net', type=str, default='resnet34')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--scheduler_step', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=.98)
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='1')
    opt = parser.parse_args()
    print(opt)

    return opt


def train(opt):
    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join('./log/', opt.log),
        filemode='w',
        format='',
        level=logging.INFO
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    # model = WSDAN(num_classes=4, M=opt.num_attentions, net=opt.net, pretrained=True).cuda()
    model = EfficientNet.from_pretrained("efficientnet-b7", advprop=True).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_step, gamma=opt.gamma)
    loss = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True

    logging.info('Network weights save to {}'.format(opt))

    data = Data('./', resize=(opt.size, opt.size))
    valdata = Data('./', phase='val', resize=(opt.size, opt.size))
    all_data = pd.read_csv('./train.csv')
    d = all_data.iloc[:, 0].values
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    targets = all_data.iloc[:, [1, 2, 3, 4]].values
    train_y = targets[:, 0] + targets[:, 1]*2+targets[:, 2]*3
    val_min = float('inf')
    train_loss = float('inf')

    for i_fold, (t, v) in enumerate(skf.split(d, train_y)):
        train_index = SubsetRandomSampler(t)
        val_index = SubsetRandomSampler(v)
        train_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch, sampler=train_index, num_workers=opt.n_cpu)
        val_loader = torch.utils.data.DataLoader(valdata, batch_size=opt.batch, sampler=val_index, num_workers=opt.n_cpu)
        with tqdm.tqdm(total=opt.epoch, miniters=1, mininterval=0) as progress:
            for epoch in range(1, opt.n_epochs+1):
                model.train()
                for i, batch in enumerate(train_loader, 1):
                    images = Variable(batch['X']).cuda()
                    correct_label = Variable(batch['Y']).cuda()
                    optimizer.zero_grad()
                    outputs = model(images)
                    train_loss = loss(outputs, correct_label)
                    train_loss.backward()
                    optimizer.step()

                    if i % 1 == 0:
                        des = "Fold: {fold}, epoch: {epoch}, Iter {iter:.1f}%, Total Loss: {loss:.4f} ".format(
                            fold=i_fold, epoch=epoch,
                            iter=100 * i/len(train_loader),
                            loss=train_loss,
                        )
                        des_loss = "lr: {lr:.5f}".format(lr=scheduler.get_lr()[0])
                        progress.set_description(des+des_loss)
                des = "Fold: {fold}, epoch: {epoch}, Total Loss: {loss:.4f} ".format(
                    fold=i_fold, epoch=epoch,
                    loss=train_loss,
                )
                des_loss = "lr: {lr:.5f}".format(lr=scheduler.get_lr()[0])

                logging.info('{}'.format(des+des_loss))

                if train_loss < train_loss:
                    train_loss = train_loss
                    torch.save(model, 'train_eff_{}.pth'.format(opt.net))
                    print('\t Train Model Saved')
                    logging.info('Train Model Saved')

                model.eval()
                correct = 0
                test_loss = 0
                for j, batch in enumerate(val_loader):
                    images = Variable(batch['X']).cuda()
                    correct_label = Variable(batch['Y']).cuda()
                    with torch.no_grad():
                        outputs = model(images)
                        y_pred = outputs
                        train_loss = loss(y_pred, correct_label)

                    test_loss += train_loss
                test_loss = 100*test_loss/len(val_index)
                des = " Test Loss: {loss:.5f}".format(
                    loss=test_loss)
                print(des)
                logging.info('{}'.format(des))

                if test_loss < val_min:
                    val_min = test_loss
                    torch.save(model, 'val_eff7.pth')
                    print('\tTest Model Saved')
                    logging.info('Test Model Saved')
                scheduler.step()


def predict(opt, model):
    testLoader = DataLoader(testData('./', resize=(opt.size, opt.size)), batch_size=1, num_workers=2)
    model = torch.load(model).cuda()
    model.eval()
    count = 0
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        for i, batch in enumerate(testLoader):
            predict_array = [batch[1][0], 0, 0, 0, 0]
            images = Variable(batch[0]).cuda()
            if count % 50 == 0:
                print(count)
            count += 1
            with torch.no_grad():
                outputs = model(images)
                y_pred = outputs
                _, label = torch.max(y_pred, 1)
                y_pred = softmax(y_pred.cpu().numpy()[0])
                for j in range(1, 5):
                    predict_array[j] = y_pred[j-1]
            writer.writerow(predict_array)


def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum()


if __name__ == '__main__':
    opt = parse()
    # train(opt)
    predict(opt, 'val_eff{}.pth'.format('7'))