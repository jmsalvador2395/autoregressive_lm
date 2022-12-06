import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from itertools import product
import datasets
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import math

# relative imports
from . import utils, data_utils, models

import pdb

def main(args = None):

	# parse arguments if none are provided
	if args is None:
		args = utils.parse_args()

	# assign dict values to variable names
	num_epochs = args['num_epochs']
	batch_size = args['batch_size']
	dev = args['device']

	# get the dataset
	data = data_utils.get_data(
		ds_name = args['dataset'],
		ds_dir = args['dataset_dir'],
	)
	
	# tokenize the dataset
	tokenized_data = data_utils.tokenize_data(
		data, 
		save_dir='/data/john/datasets/pg19_tokenized',
		keep_in_memory=args['keep_in_memory'],
	)

	# get the vocabulary for the dataset
	vocab = data_utils.build_vocab(data, min_freq=1800)

	# get dataloaders 
	dataloaders = data_utils.get_dataloaders(tokenized_data, batch_size=batch_size)
	#dataloaders = data_utils.get_dataloaders2(tokenized_data, vocab, batch_size=batch_size)

	# initialize tensorboard log
	run_name=args['model'] + '_' + str(time.time())
	writer=SummaryWriter()
	writer.add_hparams(args, {'placeholder' : 0})
	
	if args['model'] == 'lstm':
		model = models.LSTM(
			len(vocab), # vocab size
			args['model_dims'], # embedding_dim
			args['model_dims'], # hidden_dim
			args['lstm_num_layers'],
			dropout=args['dropout'], 
			tie_weights=True
		)
	elif args['model'] == 'transformer':
		#TODO implement transformer model
		pass
	
	model.to(dev)
	#model.summary((batch_size, 1))
	model.summary()

	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args['lr'],
		weight_decay=args['weight_decay']
	)

	# create a progress bar for training
	progress_bar = tqdm(
		range(num_epochs*len(dataloaders['train']))
	)
	step=0
	for epoch in range(num_epochs):
		batch_num=0
		for batch in dataloaders['train']:
			model.train()

			# get sequence lengths
			lengths = np.array([len(sample['tokens']) for sample in batch])
			seq_len = max(lengths)+1
			diffs = seq_len-lengths

			# create data to pass to the model
			int_tokens = torch.tensor(
				[vocab(sample['tokens'])+[0]*diff for sample, diff in zip(batch, diffs)],
				device=dev
			)

			# create mask
			mask = torch.zeros((int_tokens.shape)).to(torch.bool)
			for row, length in enumerate(lengths):
				mask[row, :length]=True

			# this sub-training loop needs to be used because the documents are too long
			subseq_len = seq_len if args['subseq_len'] == 0 else args['subseq_len']
			ht = torch.zeros((
				subseq_len,
				model.num_layers,
				model.hidden_dim
			)).to(dev)
			for t in range(0, seq_len-1, subseq_len):
				
				if t != 0:
					ht = (ht[0].detach(), ht[1].detach())

				# compute the submask
				submask = mask[:, 1:][:, t:t+subseq_len]

				# compute scores
				scores, ht = model(int_tokens[:, :-1][:, t:t+subseq_len], ht)
				scores = scores[torch.flatten(submask)]

				# compute labels
				labels = int_tokens[:, 1:][:, t:t+subseq_len][submask]

				# compute loss
				loss = loss_fn(scores, labels)

				# backwards pass
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()


				step+=1

				writer.add_scalar('loss/train', loss, step)
				writer.add_scalar('perplexity/train', torch.exp(loss), step)
				writer.add_scalar('bits_per_token/train', torch.log2(torch.exp(loss)), step)
				writer.add_scalar('epoch', epoch, step)
				writer.add_scalar('||w||', model.l2_norm(), step)

				# update progress bar with important tracking values
				progress_bar.set_postfix({
					'mode' : 'train',
					'batch_num' : batch_num,
					'loss/train' : loss.item(),
					'epoch' : epoch,
					'||w||' : model.l2_norm(),
				})
				progress_bar.refresh()

			batch_num+=1
			if batch_num%5 == 0:
				# evaluate on the validation set
				val_loss = evaluate(
					model,
					dataloaders['validation'],
					vocab,
					dev,
					args,
					loss_fn,
					progress_bar,
					step
				)
				
				writer.add_scalar('loss/val', val_loss, step)
				writer.add_scalar('perplexity/val', math.exp(val_loss), step)
				writer.add_scalar('bits_per_token/val', math.log2(math.exp(val_loss)), step)

			# increment the progress bar
			progress_bar.update()

	writer.close()

	import pdb
	pdb.set_trace()

def evaluate(model, dataloader, vocab, dev, args, loss_fn, progress_bar, step):
	losses=[]
	n=0
	with torch.no_grad():
		for batch in dataloader:

			# update the progress bar
			progress_bar.set_postfix({
				'mode' : 'evaluation',
				'batches_processed' : n,
			})
			progress_bar.refresh()

			# get sequence lengths
			lengths = np.array([len(sample['tokens']) for sample in batch])
			seq_len = max(lengths)+1
			diffs = seq_len-lengths

			#pdb.set_trace()

			# create data to pass to the model
			int_tokens = torch.tensor(
				[vocab(sample['tokens'])+[0]*diff for sample, diff in zip(batch, diffs)],
				device=dev
			)

			# compute mask for ignoring invalid tokens
			mask = torch.zeros((int_tokens.shape)).to(torch.bool)
			for row, length in enumerate(lengths):
				mask[row, :length]=True

			# this sub-training loop needs to be used because the documents are too long
			subseq_len = seq_len if args['subseq_len'] == 0 else args['subseq_len']
			ht = torch.zeros((
				subseq_len,
				model.num_layers,
				model.hidden_dim
			)).to(dev)
			for t in range(0, seq_len-1, subseq_len):
				
				if t != 0:
					ht = (ht[0].detach(), ht[1].detach())

				# compute the submask
				submask = mask[:, 1:][:, t:t+subseq_len]

				# compute scores
				scores, ht = model(int_tokens[:, :-1][:, t:t+subseq_len], ht)
				scores = scores[torch.flatten(submask)]

				# compute labels
				labels = int_tokens[:, 1:][:, t:t+subseq_len][submask]

				# compute loss
				loss = loss_fn(scores, labels)

				losses.append(loss.item())
			n+=1

	return np.mean(losses)
	
if __name__ == '__main__':
	main()
