import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, models

import numpy as np
import time, os, random, math, argparse
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('/content/drive/My Drive/test2/')
from ResNet import ResNet18

home_dir = '/content/drive/My Drive/test2/'


class NetUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(NetUnit, self).__init__()
		# the standard size of kernel is 3
		# padding=1: input & outputs have the same image size,
		# the size only reduces in pooling layers.
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
			kernel_size=3, stride=1, padding=1)
		"""
		a normalization layer:
		after every layer of computation, data distribution changes in unexpected way
		(called Internal Covariate Shift),
		which results in difficulties in next layer's study.
		So after every layer, normalize data.
		However, if they all follows standard Gaussian Distribution after every layer,
		it's hard for the following layers to learn any features.
		So some parameters should be reserved for study.
		what BatchNorm layer does:
		1.computes the mean value of the batch x;
		2.computes stdev of the batch x;
		3.normalize data x;
		4.import 2 variables gamma, beta: y = gamma * x + beta.
		"""
		# w/o this layer, acc remains at 0.10
		self.norm = nn.BatchNorm2d(num_features=out_channels)
		self.relu = nn.ReLU()


	def forward(self, x):
		return self.relu(self.norm(self.conv(x)))


class SimpleNet(nn.Module):
	def __init__(self, num_classes=10):
		super(SimpleNet, self).__init__()
		# 3: input has 3 channels(RGB)
		self.unit1 = NetUnit(3, 32)
		self.unit2 = NetUnit(32, 32)
		self.unit3 = NetUnit(32, 32)
		self.pool1 = nn.MaxPool2d(kernel_size=2) # 2: every 4 pixels reduce to 1
		self.unit4 = NetUnit(32, 64)
		self.unit5 = NetUnit(64, 64)
		self.unit6 = NetUnit(64, 64)
		self.unit7 = NetUnit(64, 64)
		self.pool2 = nn.MaxPool2d(kernel_size=2)
		self.unit8 = NetUnit(64, 128)
		self.unit9 = NetUnit(128, 128)
		self.unit10 = NetUnit(128, 128)
		self.unit11 = NetUnit(128, 128)
		self.pool3 = nn.MaxPool2d(kernel_size=2)
		self.unit12 = NetUnit(128, 128)
		self.unit13 = NetUnit(128, 128)
		self.unit14 = NetUnit(128, 128)
		self.pool4 = nn.AvgPool2d(kernel_size=4)

		# Sequential: add parameters to net in order
		self.head_net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, 
			self.unit4, self.unit5, self.unit6, self.unit7, self.pool2, self.unit8,
			self.unit9,self.unit10, self.unit11, self.pool3, self.unit12, self.unit13,
			self.unit14, self.pool4)
		# a full-connected layer at the end.
		# after pool1, 32*32 -> 16*16;
		# after pool2, 16*16 -> 8*8;
		# after pool3, 8*8 -> 4*4;
		# after pool4, 4*4 -> 1*1*128.
		# make the output a flat vector, so that it can be sent to FC layer.
		self.fc = nn.Linear(128, num_classes)


	def forward(self, x):
		x = self.head_net(x)
		return self.fc(x.view(-1, 128)) # output: batch * 10(num_classes)


if __name__ == '__main__':
	net = SimpleNet(10)
	print(net)
