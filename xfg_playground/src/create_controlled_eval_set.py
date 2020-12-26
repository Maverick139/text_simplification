import os
import re
import argparse
import pickle as pkl

def create_control_file(file,run_typ,res):
	print('called',file)
	data = list()
	with open(file,'r') as infile:
		for line in infile.readlines():

			prepend_toks = list()

			if res.nbchars:
				prepend_toks.append('<NbChars_{}>'.format(round(res.nbchars,2)))
			if res.levsim:
				prepend_toks.append('<LevSim_{}>'.format(round(res.levsim,2)))
			if res.wrdrank:
				prepend_toks.append('<WordRank_{}>'.format(round(res.wrdrank,2)))
			if res.deptree:
				prepend_toks.append('<DepTreeDepth_{}>'.format(round(res.deptree,2)))

			control_sentence = prepend_toks
			control_sentence.extend([line.lower()])

			data.append(' '.join(control_sentence))

	o_file = os.path.join(res.out_dir,run_typ,'wiki.{}.complex'.format(run_typ))
	print(str(o_file))
	with open(o_file,'w') as outfile:
		outfile.writelines(data)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--in-dir', action='store', dest='in_dir', type=str, default=None,
	                    help='path to raw data folder (subfolders: train, test, valid)')
	parser.add_argument('--out-dir', action='store', dest='out_dir', type=str, default=None,
	                    help='path to control data folder')
	parser.add_argument('-n','--nbchars', action='store', dest='nbchars', type=float, default=None,
	                    help='control relation value for NbChars')
	parser.add_argument('-l','--levsim', action='store', dest='levsim', type=float, default=None,
	                    help='control relation value for LevSim')
	parser.add_argument('-w','--wrdrank', action='store', dest='wrdrank', type=float, default=None,
	                    help='control relation value for WordRank')
	parser.add_argument('-d','--deptree', action='store', dest='deptree', type=float, default=None,
	                    help='control relation value for DepTreeDepth')

	res = parser.parse_args()

	try:
		os.mkdir(res.out_dir)
		os.mkdir(os.path.join(res.out_dir,'test'))
		os.mkdir(os.path.join(res.out_dir,'valid'))
		os.mkdir(os.path.join(res.out_dir,'train'))
	except:
		pass

	for run_typ in ['test']:#['test','valid','train']:
		file = os.path.join(res.in_dir,run_typ,'wiki.{}.complex'.format(run_typ))
		create_control_file(file,run_typ,res)


