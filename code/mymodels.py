import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class LSTM(nn.Module):
	def __init__(self, dim_input):

		super(LSTM, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(in_features=dim_input, out_features=64)
		self.rnn = nn.LSTM(input_size=64, hidden_size=16, num_layers=1, batch_first=True)
		self.fc2 = nn.Linear(in_features=16, out_features=2)


	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		batch_size, seq_len, num_features = seqs.size()
		x = torch.sigmoid(self.fc1(seqs))
		x = pack_padded_sequence(x, lengths, batch_first=True)
		x, _ = self.rnn(x)
		x, _ = pad_packed_sequence(x, batch_first=True, total_length=seq_len)
		y = torch.zeros(batch_size, 16).float()
		for i in range(batch_size):
			y[i,:] = x[i, lengths[i]-1, :]
		x = self.fc2(y)
		return x