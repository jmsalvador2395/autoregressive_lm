import torch
from torch import nn
import torch.nn.functional as f
import torchsummary
import math

class LSTM(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=.5, tie_weights=True):
		
		super().__init__()
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim

		# holds the word embeddings
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		
		# lstm
		self.lstm = nn.LSTM(
			embedding_dim,
			hidden_dim,
			num_layers=num_layers,
			dropout=dropout,
			batch_first=False
		)

		# dropout layer
		self.dropout = nn.Dropout(dropout)

		# classification head
		self.fc = nn.Linear(hidden_dim, vocab_size)

		if tie_weights:
			assert embedding_dim == hidden_dim, 'embedding_dim and hidden_dim don\'t match'
			self.embedding_weight = self.fc.weight

		# initialize weights
		self.init_weights()
	
	def init_weights(self):
		init_range_emb = 0.1
		init_range_other = 1/math.sqrt(self.hidden_dim)
		self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
		self.fc.weight.data.uniform_(-init_range_other, init_range_other)
		self.fc.bias.data.zero_()
		for i in range(self.num_layers):
			self.lstm.all_weights[i][0] = torch.FloatTensor(
				self.embedding_dim,
				self.hidden_dim
			).uniform_(-init_range_other, init_range_other)
			self.lstm.all_weights[i][1] = torch.FloatTensor(
				self.embedding_dim,
				self.hidden_dim
			).uniform_(-init_range_other, init_range_other)
	
	def forward(self, x, ht=None):
		outputs = []
		embedding = self.dropout(self.embedding(x))
		N, L, D = embedding.shape
		for t in range(L):
			output, ht = self.lstm(embedding[:, t, :], ht)
			output = self.dropout(output)
			outputs.append(output)

		scores = self.fc(torch.cat(outputs, dim=0))

		return scores, ht
	def summary(self, shape=None):
		torchsummary.summary(self, shape)

	def l2_norm(self):
		weights = self.parameters()
		return sum([torch.sum(w**2) for w in weights]).item()
