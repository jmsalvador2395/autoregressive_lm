import argparse

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--dataset',
		type=str,
		default='pg19',
		help='the huggingface dataset to use. (default: pg19)'
	)

	parser.add_argument(
		'--dataset_dir',
		type=str,
		default='../datasets',
		help='the directory for downloading/reading the huggingface dataset'
	)

	
	args = parser.parse_args()
	return vars(args)
