import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth_s(nn.Module):

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth_s, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		targets = targets.cuda()
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class CrossEntropyLabelSmooth_c(nn.Module):

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth_c, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		# targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		targets = targets.cuda()
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class CrossEntropy(nn.Module):

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropy, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = targets.cuda()
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class SoftEntropy(nn.Module):
	def __init__(self):
		super(SoftEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		return loss
