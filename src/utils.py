from pathlib import Path
import argparse
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchtext

def get_project_root():
	return str(Path(__file__).parent.parent)

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--dataset',
		type = str,
		default = 'pg19',
		help = 'the huggingface dataset to use. (default: pg19)'
	)

	parser.add_argument(
		'--dataset_dir',
		type = str,
		default = 'datasets',
		help = (
			'the directory for downloading/reading the huggingface dataset. ' + 
			'the input directory is relative to the root of the project'
		)
	)

	parser.add_argument(
		'--batch_size',
		type = int,
		default = 16,
		help = 'the batch size for training on the dataset'
	)

	parser.add_argument(
		'--num_epochs',
		type = int,
		default = 5,
		help = 'the number of epochs for training on the dataset'
	)

	parser.add_argument(
		'--device',
		type = str,
		default = 'cpu',
		help = 'the device for storing the model and batch data'
	)

	parser.add_argument(
		'--keep_in_memory',
		action = argparse.BooleanOptionalAction,
		default = False,
		help = 'calling this flag loads the dataset in memory'
	)

	parser.add_argument(
		'--model',
		type = str,
		choices = ['lstm', 'transformer'],
		default = 'lstm',
		help = 'this option chooses which type of model to use'
	)

	parser.add_argument(
		'--lstm_num_layers',
		type = int,
		default = 5,
		help = 'this sets the number of layers in the LSTM (default: 5)'
	)

	parser.add_argument(
		'--model_dims',
		type = int,
		default = 758,
		help = 'this sets the dimensionality of our word embeddings'
	)

	parser.add_argument(
		'--subseq_len',
		type = int,
		default = 0,
		help = (
			'this sets the length of subsequence that you want processed. ' +
			'0 means to just process the whole sequence. (default: 0)'
		)
	)

	parser.add_argument(
		'--lr',
		type = float,
		default = .01,
		help = 'this sets the learning rate for the optimizer'
	)

	parser.add_argument(
		'--weight_decay',
		type = float,
		default = 0,
		help = 'this is the regularization term'
	)
	
	parser.add_argument(
		'--dropout',
		type = float,
		default = .05,
		help = 'the dropout rate'
	)
	
	args = parser.parse_args()
	return vars(args)
