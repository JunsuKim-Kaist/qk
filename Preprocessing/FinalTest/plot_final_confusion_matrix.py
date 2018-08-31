"""
plot confusion_matrix of PublicTest and PrivateTest
"""
import sys
sys.path.insert(0, '/mnt/home/qualcomm/junsu/qk/src/')

import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
from custom import CUSTOM

from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from models import *
from openpyxl import Workbook


parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='../../src/QIA', help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
parser.add_argument('--label_blind', '-lb', action='store_true', help='if labels of test data set is not accessible')
parser.add_argument('--output_path', type=str, default='/mnt/home/qualcomm/junsu/qk/FinalTestData/TestDataset/', help='h5 file path')
parser.add_argument('--h5file_path', type=str, default='/mnt/home/qualcomm/junsu/qk/FinalTestData/TestDataset/FinalData.h5', help='h5 file path')
parser.add_argument('--Emofile_path', type=str, default='/mnt/home/qualcomm/junsu/qk/FinalTestData/TestDataset/FinalDataset.xlsx', help='Emofile path')


opt = parser.parse_args()

cut_size = 44

transform_test = transforms.Compose([
    transforms.FiveCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()

path = os.path.join(opt.dataset + '_' + opt.model)
checkpoint = torch.load(os.path.join(path, 'PrivateTest' + '_model.t7'))

net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
Testset = CUSTOM(transform=transform_test, h5file_path=opt.h5file_path, Emofile_path=opt.Emofile_path)
Testloader = torch.utils.data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0)
correct = 0
total = 0
all_target = []

wb = Workbook()
ws1 = wb.active
ws1.title = "Sheet1"

for batch_idx, (title, inputs, targets) in enumerate(Testloader):

    #print(batch_idx)


    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs, targets = inputs.cuda(), targets.cuda()
    targets = Variable(targets)
    with torch.no_grad():
        inputs = Variable(inputs)
    outputs = net(inputs)

    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
    _, predicted = torch.max(outputs_avg.data, 1)

    ws1["A"+str(batch_idx+1)] = title[0]
    ws1["B"+str(batch_idx+1)] = class_names[predicted]
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()
    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted),0)
        all_targets = torch.cat((all_targets, targets),0)

wb.save(opt.output_path + "result.xlsx")

if not opt.label_blind:
    acc = 100. * correct / total
    print("accuracy: %0.3f" % acc)

    # Compute confusion matrix

    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                          title= "CustomTest"+' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
    plt.savefig(os.path.join(path, "CustomTest" + '_cm.png'))
    plt.close()
