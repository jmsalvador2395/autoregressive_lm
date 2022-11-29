import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from itertools import product
import datasets
from datasets import load_dataset
from tqdm import tqdm

# relative imports
from . import utils, data_utils, models

def main(args = None):

	# parse arguments if none are provided
	if args is None:
		args = utils.parse_args()

	# assign dict values to variable names
	num_epochs = args['num_epochs']
	batch_size = args['batch_size']

	# TODO define the models

	data, dataloaders = data_utils.get_data(
		ds_name = args['dataset'],
		ds_dir = args['dataset_dir'],
		batch_size = batch_size
	)
	
	# create a progress bar for training
	progress_bar = tqdm(
		range(num_epochs*len(dataloaders['train']))
	)
	
	print('tokenizing')
	tokenized_data = data_utils.build_vocab(data)
	print('done')
	return
	for epoch in range(args['num_epochs']):
		for batch in dataloaders['train']:
			# increment the progress bar
			progress_bar.update()
			print(batch['text'][0])

			# TODO make training code

			# update progress bar with important tracking values
			progress_bar.set_postfix({
				'test' : 10,
			})

	
if __name__ == '__main__':
	main()
