import os
import re
import argparse
import pickle as pkl
from custom_similarity_metric import *

def file_eval(fpath,param,silent=False):

	# option parsing
	filename = fpath	# fsq_gen_(test/valid)_(rel_code)_ep(i).txt
	filename = filename.split('_')
	run_typ = filename[2]
	rel_code = filename[3]
	epoch = filename[4]

	rel_token_count = int(rel_code.count('1'))
	probablity_score_offset = 1

	i = 0
	data = list()
	s, t, h = dict(), dict(), dict()
	with open(os.path.join(param.dir_path,fpath)) as infile:
		for line in infile.readlines():
			line = re.sub('\t',' ',line.rstrip('\n')).split()

			prefix = line[0].split('-')

			if prefix[0]=='S':	# input sentence
				s[int(prefix[1])] = line[1+rel_token_count:].copy()
				h[int(prefix[1])] = line[1+rel_token_count:].copy()

			elif prefix[0]=='T':	# target sentence
				t[int(prefix[1])] = line[1:]

			elif prefix[0]=='H':	# target sentence
				pass
				#h[int(prefix[1])] = line[1+probablity_score_offset:]

	n = max(list(s.keys()))

	for i in range(1,n+1):
		data.append([s[i],t[i],h[i]])

	sari_score = 0
	for i in range(len(data)):
		s_score,res = SARI4(*data[i])
		sari_score += s_score/0.7559

	sari_cumm = sari_score/len(data)
	bleu_cumm = None

	if not silent:
		print('>>> evaluation complete: {}'.format(str(os.path.join(param.dir_path,fpath))))
		print('       SARI4: {} | BLEU4: {}'.format(round(sari_cumm,2),round(0,2)))

	return sari_cumm, bleu_cumm

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-g','--gen-folder', action='store', dest='dir_path', type=str, default=None,
	                    help='path to fairseq generate folder')
	parser.add_argument('--sari', action='store_true', dest='sari', default=True,
	                    help='generate cummulative SARI4 score')
	parser.add_argument('--bleu', action='store_true', dest='bleu', default=False,
	                    help='generate cummulative BLEU4 score')
	parser.add_argument('--record', action='store', dest='record', default=None,
	                    help='path to pickle file to save epoch performance record in py3 dict()')

	param = parser.parse_args()

	if not param.dir_path:
		print('>>> ERROR: no generate file passed (--gen-file)')
		exit()

	for file in os.listdir(param.dir_path):
		bleu_cumm, sari_cumm = file_eval(file,param,silent=False)

		# full_path = os.path.join(param.dir_path,file)
		# if param.record:

		# 	record = None

		# 	if not os.path.exists(full_path):
		# 		_ = open(full_path,'wb')
		# 		record = {'meta':{'run_typ':None,'rel_code':None},
		# 					'epoch':{}}

		# 	else:
		# 		with open(full_path,'rb') as infile:
		# 			record = pkl.load(infile)

		# 	filename = file.split('_')
		# 	run_typ = filename[2]
		# 	rel_code = filename[3]
		# 	epoch = filename[4]

		# 	record[]
		# 	with open(full_path,'wb') as outfile:

			
