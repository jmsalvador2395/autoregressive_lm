from pathlib import Path
import argparse
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchtext
from torchtext.vocab import vocab
import os
import pickle
from tqdm import tqdm, std
from collections import Counter
from nltk.tokenize import RegexpTokenizer

# relative imports
from . import utils

def collate_fn(batch):
	"""
	collate function for the dataloader. for some reason the dataloader 
	wraps each token with a tuple if i don't do this so whatever i guess
	"""
	return batch

def get_data(ds_name='pg19', ds_dir='datasets', batch_size=16):

	# build full directory path
	root_dir = utils.get_project_root()
	target_dir = root_dir + '/' + ds_dir

	# create path for dataset if it doesn't exist
	path = Path(target_dir).mkdir(parents=True, exist_ok=True)
	datasets.config.DOWNLOADED_DATASETS_PATH = path
	
	# download or read in dataset
	data = load_dataset(
		ds_name,
		cache_dir = target_dir
	).with_format('torch')

	return data

def get_dataloaders(data, batch_size=16):

	# create dictionary of dataloaders depending on the splits
	splits=list(data.keys())
	dataloaders={
		split : DataLoader(
			data[split], 
			batch_size=batch_size, 
			collate_fn=collate_fn,
			shuffle=True
		) for split in splits
	}

	return dataloaders

def get_dataloaders2(data, vocab, batch_size=16):

	print('building custom dataloaders...')
	splits = data.keys()
	dataloaders = {split:[] for split in splits}
	import pdb
	bar = tqdm(range(
		len(splits) +
		sum([len(data[split]) for split in splits])
	))
	for split in splits:
		for sample in data[split]:
			if sample['tokens']:
				sample['tokens'].append('<eos>')
				tokens = vocab(sample['tokens'])
				dataloaders[split].append(tokens)
			bar.set_postfix({'current split' : split})
			bar.update()
	print('done')

	pdb.set_trace()

def tokenize_data(data, ds_name='pg19', save_dir=None, keep_in_memory=False):
	"""
	tokenizes a given dataset

	input:
		data : the huggingface dataset
		ds_name : name of the dataset
		save_dir : the save directory for the dataset (default: <project root>/datasets/tokenized/)
	
	output:
		the tokenized dataset
	"""
	# use default directory if none is provided
	if save_dir is None:
		save_dir = utils.get_project_root() + f'/datasets/tokenized/{ds_name}'

	# check if the tokenized dataset exists and load. otherwise proceed with tokenization
	if os.path.exists(save_dir):
		tokenized_data = datasets.load_from_disk(
			save_dir,
			keep_in_memory=keep_in_memory
		)
		return tokenized_data.with_format('torch')
	
	# make sure path exists
	path = Path(save_dir).mkdir(parents=True, exist_ok=True)

	# tokenize the dataset
	# source: https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf
	print('creating tokenized dataset (this only needs to happen once per dataset)...')
	#tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
	#tokenize = lambda sample, tokenizer: {'tokens' : tokenizer(sample['text'])}
	tokenizer = RegexpTokenizer(r'\w+')
	tokenize = lambda sample, tokenizer: {'tokens' : tokenizer.tokenize(sample['text'].lower())}
	tokenized_data = data.map(
		tokenize,
		remove_columns=['text'],
		fn_kwargs={'tokenizer' : tokenizer}
	)

	# save the dataset
	tokenized_data.save_to_disk(save_dir)
	print(f'tokenized dataset saved to {save_dir + ds_name}')

	return tokenized_data

def build_vocab(data, ds_name='pg19', token_ds_dir=None, 
				vocab_dir=None, min_freq=100, pretokenized=True,
				mem_hog=False):
	"""
	builds the vocabulary for the dataset
	source: https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf

	input:
		data : the huggingface dataset
		ds_name : name of the dataset
		save_dir : the save directory for the dataset (default: <project root>/datasets/tokenized/)
	
	output:
		the dataset vocabulary
	"""

	if vocab_dir is None:
		vocab_dir = utils.get_project_root() + f'/datasets/tokenized/{ds_name}'
		vocab_path = vocab_dir + '/vocab.pkl'
	
	# if vocab already exists, load and return
	if os.path.isfile(vocab_path):
		with open(vocab_path, 'rb') as f:
			vocab = pickle.load(f)
		return vocab

	# get the tokenized dataset
	if not pretokenized:
		tokenized_data = tokenize_data(data, ds_name=ds_name, save_dir=token_ds_dir)

	# build vocab
	print(
		'building vocabulary ' +
		'(this takes an obscene amount of memory so it gets saved after)...'
	)

	if not mem_hog:
		freqs = {}
		bar = tqdm(tokenized_data['train'])
		for sample in bar:
			
			for token in sample:
				if token in freqs:
					freqs[token] += 1
				else:
					freqs[token] = 1

			bar.set_postfix({
				'dict size' : len(freqs)
			})
			bar.update()

		vocab = torchtext.vocab.vocab(freqs, min_freq=min_freq)
	else:
		vocab = torchtext.vocab.build_vocab_from_iterator(
			tokenized_data['train']['tokens'],
			min_freq=min_freq
		)
	vocab.insert_token('<unk>', 0)
	vocab.insert_token('<eos>', 1)
	vocab.set_default_index(vocab['<unk>'])

	print('done')

	# save the vocab
	print(f'saving the vocab to {vocab_path}')
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)
	print('done')

	return vocab
