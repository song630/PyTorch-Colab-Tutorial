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


def parse_args():
	parser = argparse.ArgumentParser(description='Train a network')
	# it should achieve an accuracy of more than 90% after 35 epoches
	# original: 40
	parser.add_argument('--epochs', dest='max_epochs',
						help='number of epochs to train',
						default=40, type=int)
	parser.add_argument('--nw', dest='num_workers',
						help='number of worker to load data', # ??
						default=0, type=int)
	# original: 32
	parser.add_argument('--crop', dest='crop_size',
						help='crop size',
						default=32, type=int)
	parser.add_argument('--cuda', dest='cuda',
						help='whether use CUDA',
						action='store_true')
	# original: 10
	parser.add_argument('--decay', dest='decay_step',
						help='decay step',
						default=10, type=int)
	parser.add_argument('--net', dest='net_type',
						help='use SimpleNet or ResNet18',
						default='SimpleNet', type=str)
	"""
	parser.add_argument('--mGPUs', dest='mGPUs', # === ?? try this later
						help='whether use multiple GPUs',
						action='store_true')
	"""
	# original: 32
	parser.add_argument('--bs', dest='batch_size',
						help='batch_size',
						default=32, type=int)
	# original: 1e-3
	parser.add_argument('--lr', dest='lr',
						help='starting learning rate',
						default=1e-3, type=float)
	parser.add_argument('--r', dest='resume',
						help='resume checkpoint or not',
						default=False, type=bool)
	parser.add_argument('--checkepoch', dest='checkepoch',
						help='checkepoch to load model',
						default=1, type=int)
	parser.add_argument('--save_dir', dest='save_dir',
						help='directory to save models', default="/content/drive/My Drive/test2/load/",
						type=str)
	return parser.parse_args()


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


def TrainTransform(image, crop_size=32):
	preprocess = tf.Compose([
		tf.RandomHorizontalFlip(),
		tf.RandomCrop(crop_size, padding=4), # ??
		tf.ToTensor(),
		# make the values of all pixels between (-1, 1)
		tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # ??
	])
	return preprocess(image)


def TestTransform(image):
	# when test, there's no batch, so no need to crop
	preprocess = tf.Compose([
		tf.ToTensor(),
		tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # ??
	])
	return preprocess(image)


def test():
	test_total = 10000 # 10,000 images in test set
	test_acc = 0.0
	net.eval() # net: from outer scope
	for inputs, targets in testloader: # testloader: from outer scope
		if args.cuda:
			inputs = Variable(inputs.cuda())
			targets = Variable(targets.cuda())
		else:
			raise Exception('Should use cuda here.')
		# forward
		outputs = net(inputs)
		_, prediction = torch.max(outputs.data, dim=1)
		test_acc += torch.sum(prediction == targets.data)
	test_acc /= test_total
	return test_acc


if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	input_size = args.crop_size # 32
	train_total = 50000 # 50,000 images in training set
	# training images: 50,000
	trainset = torchvision.datasets.CIFAR10(home_dir + 'data/',
		train=True, download=True, transform=TrainTransform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
		shuffle=True, num_workers=args.num_workers)
	testset = torchvision.datasets.CIFAR10(home_dir + 'data/',
		train=False, download=True, transform=TestTransform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size*2,
		shuffle=True, num_workers=args.num_workers)

	num_classes = 10
	if args.net_type == 'SimpleNet':
		net = SimpleNet(num_classes) # === try to replace it with ResNet-18
	elif args.net_type == 'ResNet18':
		net = ResNet18()
	else:
		raise Exception('Invalid input net type.')
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

	if args.cuda:
		net.cuda()
	else:
		raise Exception('Should use cuda here.')

	start_epoch = 0

	if args.resume:
		load_name = args.save_dir + 'SimpleNet_{}.pth'.format(args.checkepoch)
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		start_epoch = checkpoint['epoch']
		net.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		# load lr of last time:
		#adjust_learning_rate(optimizer, optimizer.param_groups[0]['lr'])
		print("loaded checkpoint %s" % (load_name))

	for epoch in range(start_epoch, args.max_epochs):
		start = time.time()
		train_loss, train_acc, best_acc = 0.0, 0.0, 0.0
		scheduler.step() # CHECKED.
		net.train()
		for inputs, targets in trainloader: # iterations in an epoch
			if args.cuda:
				inputs = Variable(inputs.cuda())
				targets = Variable(targets.cuda())
			else:
				raise Exception('Should use cuda here.')
			# forward
			outputs = net(inputs)
			loss = criterion(outputs, targets) # compute loss
			# backward
			optimizer.zero_grad() # clear grad
			loss.backward() # propagate computed loss
			optimizer.step()
			#print('loss:', loss.cpu().data)
			train_loss += loss.cpu().data * inputs.size(0)
			_, prediction = torch.max(outputs.data, dim=1)
			# get sum of the correct predictions within a batch
			train_acc += torch.sum(prediction == targets.data)
		# when finishing an epoch
		train_loss /= train_total
		train_acc /= train_total
		test_acc = test()
		end = time.time()
		print('after epoch {}/{}, train_loss: {:.3f}, train_acc: {:.2f}, test_acc: {:.2f}, time: {:.2f}'.format(
			epoch + 1, args.max_epochs, train_loss, train_acc, test_acc, end - start))
		if test_acc > best_acc: # then update
			best_acc = test_acc
			# save the current state:
			save_name = args.save_dir + 'SimpleNet_{}.pth'.format(epoch)
			torch.save({
				'epoch': epoch + 1,
				'model': net.state_dict(),
				'optimizer': optimizer.state_dict()},
				save_name
			)
			print('updated, epoch {} saved'.format(epoch + 1))
