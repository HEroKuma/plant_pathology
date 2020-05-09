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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--beta', type=float, default=5e-2)
    parser.add_argument('--num_attentions', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log', type=str, default='efficientnet-b1.log')
    parser.add_argument('--net', type=str, default='efficientnet-b1')
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
    model = WSDAN(num_classes=4, M=opt.num_attentions, net=opt.net, pretrained=True).cuda()
    feature_center = torch.zeros(4, opt.num_attentions * model.num_features).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_step, gamma=opt.gamma)
    loss = nn.CrossEntropyLoss()
    center_loss = CenterLoss()
    torch.backends.cudnn.benchmark = True

    if opt.ckpt is not None:
        checkpoint = torch.load(opt.ckpt)

        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(opt.ckpt))

        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            logging.info('feature_center loaded from {}'.format(opt.ckpt))

    logging.info('Network weights save to {}'.format(opt))

    loss_container = AverageMeter(name='loss')
    raw_metric = TopKAccuracyMetric(topk=(1, 4))
    crop_metric = TopKAccuracyMetric(topk=(1, 4))
    drop_metric = TopKAccuracyMetric(topk=(1, 4))

    data = Data('./')
    valdata = Data('./', phase='val')
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
                raw_rate = 0
                crop_rate = 0
                drop_rate = 0
                for i, batch in enumerate(train_loader, 1):
                    loss_container.reset()
                    raw_metric.reset()
                    crop_metric.reset()
                    drop_metric.reset()

                    images = Variable(batch['X']).cuda()
                    correct_label = Variable(batch['Y']).cuda()
                    optimizer.zero_grad()
                    outputs, feature_matrix, attention_map = model(images)

                    feature_center_batch = F.normalize(feature_center[correct_label], dim=-1)
                    feature_center[correct_label] += opt.beta * (feature_matrix.detach() - feature_center_batch)

                    # Attention Cropping
                    with torch.no_grad():
                        crop_images = batch_augment(images, attention_map[:, :1, :, :], mode='crop', theta=(.5, .6), padding_ratio=.1)

                    y_pred_crop, _, _ = model(crop_images)

                    # Attention Dropping
                    with torch.no_grad():
                        drop_images = batch_augment(images, attention_map[:, :1, :, :], mode='drop', theta=(.3, .5))

                    y_pred_drop, _, _ = model(drop_images)

                    batch_loss = loss(outputs, correct_label)/3. + loss(y_pred_crop, correct_label)/3.\
                    + loss(y_pred_drop, correct_label)/3. + center_loss(feature_matrix, feature_center_batch)

                    # loss = criterion(outputs, correct_label)
                    batch_loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        epoch_loss = loss_container(batch_loss.item())
                        epoch_raw_acc = raw_metric(outputs, correct_label)
                        epoch_crop_acc = crop_metric(y_pred_crop, correct_label)
                        epoch_drop_acc = drop_metric(y_pred_drop, correct_label)

                    raw_rate = raw_rate + epoch_raw_acc[0]
                    crop_rate = crop_rate + epoch_crop_acc[0]
                    drop_rate = drop_rate + epoch_drop_acc[0]

                    if i % 1 == 0:
                        des = "Fold: {fold}, epoch: {epoch}, Iter {iter:.1f}%, Total Loss: {loss:.4f} ".format(
                            fold=i_fold, epoch=epoch,
                            iter=100 * i/len(train_loader),
                            loss=epoch_loss,
                        )
                        # des_loss = "Raw Acc: {r1:.2f}, Crop Acc: {c1:.2f}, Drop Acc: {d1:.2f}, lr: {lr:.5f}".format(
                        #                                             lr=scheduler.get_lr()[0], r1=epoch_raw_acc[0],
                        #                                             c1=epoch_crop_acc[0], d1=epoch_drop_acc[0])
                        des_loss = "Raw Acc: {r1:.2f}, Crop Acc: {c1:.2f}, Drop Acc: {d1:.2f}, lr: {lr:.5f}".format(
                                                                    lr=scheduler.get_lr()[0], r1=raw_rate/len(train_loader),
                                                                    c1=crop_rate/len(train_loader), d1=drop_rate/len(train_loader))
                        progress.set_description(des+des_loss)
                des = "Fold: {fold}, epoch: {epoch}, Total Loss: {loss:.4f} ".format(
                    fold=i_fold, epoch=epoch,
                    loss=epoch_loss,
                )
                des_loss = "Raw Acc: {r1:.2f}, Crop Acc: {c1:.2f}, Drop Acc: {d1:.2f}, lr: {lr:.5f}".format(
                    lr=scheduler.get_lr()[0], r1=epoch_raw_acc[0],
                    c1=epoch_crop_acc[0], d1=epoch_drop_acc[0])

                logging.info('{}'.format(des+des_loss))

                if epoch_loss < train_loss:
                    train_loss = epoch_loss
                    torch.save(model, 'train_wsdan_{}.pth'.format(opt.net))
                    print('\t Train Model Saved')
                    logging.info('Train Model Saved')

                loss_container.reset()
                raw_metric.reset()
                model.eval()
                correct = 0
                test_loss = 0
                for j, batch in enumerate(val_loader):
                    images = Variable(batch['X']).cuda()
                    correct_label = Variable(batch['Y']).cuda()
                    with torch.no_grad():
                        outputs, _, attention_map = model(images)
                        crop_images = batch_augment(images, attention_map, mode='crop', theta=.1, padding_ratio=.05)
                        y_pred_crop, _, _ = model(crop_images)
                        y_pred = (outputs + y_pred_crop) / 2
                        batch_loss = loss(y_pred, correct_label)
                        epoch_loss = loss_container(batch_loss.item())

                        epoch_acc = raw_metric(y_pred, correct_label)
                    test_loss += epoch_loss
                test_loss = 100*test_loss/len(val_index)
                des = " Test Loss: {loss:.5f}, acc: {a1:.2f}".format(
                    loss=test_loss,
                    a1=epoch_acc[0])
                print(des)
                logging.info('{}'.format(des))

                if test_loss < val_min:
                    val_min = test_loss
                    torch.save(model, 'val_wsdan_{}.pth'.format(opt.net))
                    print('\tTest Model Saved')
                    logging.info('Test Model Saved')
                scheduler.step()


def predict(model):
    testLoader = DataLoader(testData('./'), batch_size=1, num_workers=2)
    model = torch.load(model).cuda()
    model.eval()
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        for i, batch in enumerate(testLoader):
            predict_array = [batch[1][0], 0, 0, 0, 0]
            images = Variable(batch[0]).cuda()
            with torch.no_grad():
                outputs, _, attention_map = model(images)
                crop_images = batch_augment(images, attention_map, mode='crop', theta=.1, padding_ratio=.05)
                y_pred_crop, _, _ = model(crop_images)
                y_pred = (outputs + y_pred_crop)/2
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
    train(opt)
    predict('val_wsdan_{}.pth'.format('inception_mixed_6e'))