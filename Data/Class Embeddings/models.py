import torch
import torch.nn as nn


def weights_init(m):
	"""Initializes weights of a neural network
	"""
	for p in m.parameters():
		nn.init.normal_(p, mean=0.0, std=0.02)


class GRU(nn.Module):
	def __init__(self, emd_size, hidden_size, num_cls):
		super(GRU, self).__init__()
		self.hidden_size = hidden_size
		#self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(emd_size, hidden_size)
		self.linear_cls = nn.Linear(hidden_size, num_cls)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden0):
		#embedded = self.embedding(input).view(len(input), 1, -1)
		_, last_out = self.gru(input, hidden0)
		out_cls = self.linear_cls(last_out)
		return self.logsoftmax(out_cls.squeeze(0)), last_out


class LSTM(nn.Module):
	def __init__(self, emb_size, hidden_size, num_cls):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		#self.embedding = nn.Embedding(input_size, hidden_size)
		self.lstm = nn.LSTM(emb_size, hidden_size)
		self.linear_cls = nn.Linear(hidden_size, num_cls)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden0):
		#embedded = self.embedding(input).view(len(input), 1, -1)
		_, (last_out, _) = self.lstm(input, (hidden0, hidden0))
		out_cls = self.linear_cls(last_out)
		return self.logsoftmax(out_cls.squeeze(0)), last_out
		