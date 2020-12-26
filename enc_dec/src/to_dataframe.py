import pandas as pd
import argparse
from tqdm import tqdm
import pickle as pkl
from os import path

def file2list(fpath):
	try:
		flist = list()
		with open(fpath,'r') as infile:
			print('>>> reading file: {}'.format(fpath))
			for line in tqdm(infile.readlines()):
				flist.append(line.rstrip('\n'))
			return True, flist
	except:
		return False, None

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--complex', action='store', dest='fComp', type=str, default=None,
	                    help='path to complex corpus')
	parser.add_argument('-s','--simple', action='store', dest='fSimp', type=str, default=None,
	                    help='path to simple corpus')
	parser.add_argument('-o','--out', action='store', dest='fOut', type=str, default=None,
	                    help='path to save output parallel corpus pickle')

	param = parser.parse_args()

	res, complexData = file2list(param.fComp)
	if not res:
		print('>>> reading failed')
		exit()
	res, simpleData = file2list(param.fSimp)
	if not res:
		print('>>> reading failed')
		exit()

	parallelData = pd.DataFrame({'complex':complexData,'simple':simpleData})

	with open(param.fOut,'wb') as outfile:
		pkl.dump(parallelData,outfile)
		print('>>> written parallel corpus to file: {}'.format(param.fOut))