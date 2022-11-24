from argparser import *
from models import *

import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from itertools import product
import datasets
from datasets import load_dataset
from tqdm import tqdm



def get_dataloaders(ds_name='pg19', ds_dir='../datasets'):

	# create path for dataset if it doesn't exist
	Path(ds_dir).mkdir(parents=True, exist_ok=True)
	
	# download or read in dataset
	ds = load_dataset(
		ds_name,
		data_dir=ds_dir,
	).with_format('torch')

	# create dictionary of dataloaders depending on the splits
	splits=list(ds.keys())
	dataloaders=[DataLoader(ds[split]) for split in splits]

	return dataloaders

def main():
	args = parse_args()
	dataloaders = get_dataloaders(
		ds_name = args['dataset'],
		ds_dir = args['dataset_dir']
	)
	
if __name__ == '__main__':
	main()
