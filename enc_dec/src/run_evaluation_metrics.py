import pandas as pd
import numpy as np
import Levenshtein
import spacy
import pickle as pkl
from nltk.corpus import stopwords
import os
import re
import argparse
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from custom_similarity_metric import *

# 1. character ratio (Compression ratio)
###########################################################################################
def characterRatio(comp,simp):
    if len(comp) == 0:
        print('>>> character ratio: zero-exception')
        return 0
    compScore = 0 
    simpScore = 0
    for w in comp:
        compScore+=len(w)
    for w in simp:
        simpScore+=len(w)
    return simpScore/compScore

# 2. levenshtein similarity (Paraphrase ratio)
###########################################################################################

def levenshtein_similarity(comp, simp):
    '''
    calculate levenshtein similarity between aligned complex and simple  sentences
    '''
    return Levenshtein.ratio(comp, simp)

# 3. word rank (Lexical Complexity ratio)
###########################################################################################

def buildRank(fpath):
    logRank = dict()
    with open(fpath,'rb') as infile:
        wordRank = pkl.load(infile)
        for w in wordRank:
            logRank[w] = np.log(wordRank[w]+1)
    return logRank

def lexical_complexity(sentence,q=0.75):
    '''
        qth quantile of log rank of contituent words
    '''
    sentence = [w for w in sentence if w not in stopwords and w in logRank]
    if len(sentence) == 0:
        return 1
    else:
        logSentence = [logRank[w] for w in sentence]
        return np.quantile(logSentence,q)

# 4. dependency depth (Syntactical Complexity ratio)
###########################################################################################

# def prep_spacy_model(model):
#     '''load spacy's pretrained english model'''
#     if not spacy.util.is_package(model):
#         spacy.cli.download(model)
#         spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
#     return spacy.load(model)

def subtree_depth(node):
    '''
    helper to find depth from a given node of dependency tree
    '''
    if len(list(node.children)) == 0:
        return 0
    return 1 + max([subtree_depth(child) for child in node.children])

def dependency_tree_depth(sent):
    '''
    obtain dependency tree of sentence using spacy parser, and find the max depth of that tree
    '''
    tree = PARSER(sent)
    depths = [subtree_depth(sent.root) for sent in tree.sents]
    return max(depths)    

def init_spacy_model(arch):
    return spacy.load(arch)

###########################################################################################

print('>>> initializing run')
stopwords = stopwords.words('english')
PARSER = init_spacy_model('en_core_web_sm')
logRank = buildRank('src/FastTextWordRank.pkl')

###########################################################################################

rawTestSent = list()
with open('data/raw_data/test/wiki.test.complex','r') as infile:
	for line in infile.readlines():
		line = line.lower().rstrip('\n').split()
		rawTestSent.append(line)

rawValidSent = list()
with open('data/raw_data/valid/wiki.valid.complex','r') as infile:
	for line in infile.readlines():
		line = line.lower().rstrip('\n').split()
		rawValidSent.append(line)

modelTokCount = {'model1':3,'model2':0,'model3':4,'model4':1,'model5':1,'model6':1,
				'model7':1,'model8':1,'model9':1,'model10':1,'model11':1,'model12':2,
				'model13':2,'model14':2,'model15':2,'model16':2,'model17':2,'model18':2}

###########################################################################################

TEST_SENT_COUNT = 359
VALID_SENT_COUNT = 992
EPOCH_COUNT = 50
MODEL_COUNT = 18

coverage = 0

# stypDict = {'S':None,'T':None,'H':None}
# testSentDictR = {i:{'S':None,'T':None,'H':None} for i in range(TEST_SENT_COUNT)}
# validSentDictR = {i:{'S':None,'T':None,'H':None} for i in range(VALID_SENT_COUNT)}
# testEpochDictR = {'epoch{}'.format(j+1):{k:{'S':None,'T':None,'H':None} for k in range(TEST_SENT_COUNT)} for j in range(EPOCH_COUNT)}
# validEpochDictR = {'epoch{}'.format(j+1):{k:{'S':None,'T':None,'H':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)}

record = {'model{}'.format(i+1):{'eval':{'epoch{}'.format(j+1):{k:{'S':None,'T':None,'H':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)},'val':{'epoch{}'.format(j+1):{k:{'S':None,'T':None,'H':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)}} for i in range(MODEL_COUNT)}
# record = {'model{}'.format(i+1):{'eval':testEpochDictR.copy(),'val':validEpochDictR.copy()} for i in range(MODEL_COUNT)}

# scoreDict = {'SARI4':None,'BLEU4':None}
# testSentDictS = {i:{'SARI4':None,'BLEU4':None} for i in range(TEST_SENT_COUNT)}
# validSentDictS = {i:{'SARI4':None,'BLEU4':None} for i in range(VALID_SENT_COUNT)}
# testEpochDictS = {'epoch{}'.format(j+1):{k:{'SARI4':None,'BLEU4':None} for k in range(TEST_SENT_COUNT)} for j in range(EPOCH_COUNT)}
# validEpochDictS = {'epoch{}'.format(j+1):{k:{'SARI4':None,'BLEU4':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)}

score = {'model{}'.format(i+1):{'eval':{'epoch{}'.format(j+1):{k:{'SARI4':None,'BLEU4':None} for k in range(TEST_SENT_COUNT)} for j in range(EPOCH_COUNT)},'val':{'epoch{}'.format(j+1):{k:{'SARI4':None,'BLEU4':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)}} for i in range(MODEL_COUNT)}
# score = {'model{}'.format(i+1):{'eval':testEpochDictS.copy(),'val':validEpochDictS.copy()} for i in range(MODEL_COUNT)}

# tokenDict = {'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None}
# testSentDictT = {i:{'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None} for i in range(TEST_SENT_COUNT)}
# validSentDictT = {i:{'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None} for i in range(VALID_SENT_COUNT)}
# testEpochDictT = {'epoch{}'.format(j+1):{k:{'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None} for k in range(TEST_SENT_COUNT)} for j in range(EPOCH_COUNT)}
# validEpochDictT = {'epoch{}'.format(j+1):{k:{'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)}

tokens = {'model{}'.format(i+1):{'eval':{'epoch{}'.format(j+1):{k:{'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None} for k in range(TEST_SENT_COUNT)} for j in range(EPOCH_COUNT)},'val':{'epoch{}'.format(j+1):{k:{'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None} for k in range(VALID_SENT_COUNT)} for j in range(EPOCH_COUNT)}} for i in range(MODEL_COUNT)}
# tokens = {'model{}'.format(i+1):{'eval':testEpochDictT.copy(),'val':validEpochDictT.copy()} for i in range(MODEL_COUNT)}

print('>>> extracting sentences')

for i in tqdm(range(MODEL_COUNT)):

	model = 'model{}'.format(i+1)

	for phase in ['eval','val']:

		dirname = 'models/model{}/{}'.format(i+1,phase)
		# print('\n{}\n'.format(dirname))

		for fname in os.listdir(dirname):

			epoch = 'epoch'+str(fname.split('_')[-1].split('.')[0])

			with open(os.path.join(dirname,fname),'r') as infile:

				for line in infile.readlines():
					line = re.sub('\t',' ',line.rstrip('\n')).split()
					
					prefix = line[0].split('-')
					if len(prefix) != 2:
						continue
					if prefix[0] not in ['S','T','H']:
						continue

					coverage += 1	# valid entry	
					styp, sid = prefix
					sid = int(sid)

					if styp == 'S':
						record[model][phase][epoch][sid]['S'] = line[1+modelTokCount[model]:]
					elif styp == 'T':
						record[model][phase][epoch][sid]['T'] = line[1:]
					elif styp == 'H':
						# if sid == 1:
						# 	print(line)
						record[model][phase][epoch][sid]['H'] = line[2:]

	# print(record[model]['val']['epoch40'][1]['H'])

print('>>> saving extracted sentences')
with open('analysis/record.pkl','wb') as outfile:
	pkl.dump(record,outfile)

print('>>> calculating eval scores')

phase = 'eval'
for i in range(MODEL_COUNT):
	model = 'model{}'.format(i+1)
	dirname = 'models/model{}/{}'.format(i+1,phase)
	print('>>> evaluating {}'.format(model))
	for fname in os.listdir(dirname):
		epoch = 'epoch'+str(fname.split('_')[-1].split('.')[0])
		print('>>> {}'.format(epoch))
		for sid in tqdm(range(TEST_SENT_COUNT)):

			temp = record[model][phase][epoch][sid]
			data = [temp['S'],temp['T'],temp['H']]

			s_score,res = SARI4(*data)
			score[model][phase][epoch][sid]['SARI4'] = s_score

			# print(s_score)

			comp, simp = ' '.join(rawTestSent[sid]), ' '.join(temp['H'])
			tokenDict = {'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None}

			#1
			tokenDict['NbChars'] = characterRatio(rawTestSent[sid],temp['H'])

			#2
			tokenDict['LevSim'] = levenshtein_similarity(comp,simp)

			#3
			rankComp = lexical_complexity(rawTestSent[sid])
			rankSimp = lexical_complexity(temp['H'])
			tokenDict['WordRank'] = rankSimp/rankComp if rankComp>0 else 0

			#4
			depComp, depSimp = dependency_tree_depth(comp), dependency_tree_depth(simp)
			tokenDict['DepTreeDepth'] = depSimp/depComp if depComp>0 else 0

			# print(tokenDict)

			tokens[model][phase][epoch][sid] = tokenDict

	with open('analysis/tokens.pkl','wb') as outfile:
		pkl.dump(tokens,outfile)
	with open('analysis/score.pkl','wb') as outfile:
		pkl.dump(score,outfile)

print('>>> calculating val scores')

phase = 'val'
for i in range(MODEL_COUNT):
	model = 'model{}'.format(i+1)
	dirname = 'models/model{}/{}'.format(i+1,phase)
	print('>>> evaluating {}'.format(model))
	for fname in os.listdir(dirname):
		epoch = 'epoch'+str(fname.split('_')[-1].split('.')[0])
		print('>>> {}'.format(epoch))
		for sid in tqdm(range(VALID_SENT_COUNT)):

			temp = record[model][phase][epoch][sid]
			data = [temp['S'],temp['T'],temp['H']]

			s_score,res = SARI4(*data)
			score[model][phase][epoch][sid]['SARI4'] = s_score

			# print(s_score)

			comp, simp = ' '.join(rawValidSent[sid]), ' '.join(temp['H'])
			tokenDict = {'NbChars':None,'LevSim':None,'WordRank':None,'DepTreeDepth':None}

			#1
			tokenDict['NbChars'] = characterRatio(rawValidSent[sid],temp['H'])

			#2
			tokenDict['LevSim'] = levenshtein_similarity(comp,simp)

			#3
			rankComp = lexical_complexity(rawValidSent[sid])
			rankSimp = lexical_complexity(temp['H'])
			tokenDict['WordRank'] = rankSimp/rankComp if rankComp>0 else 0

			#4
			depComp, depSimp = dependency_tree_depth(comp), dependency_tree_depth(simp)
			tokenDict['DepTreeDepth'] = depSimp/depComp if depComp>0 else 0

			# print(tokenDict)

			tokens[model][phase][epoch][sid] = tokenDict

	with open('analysis/tokens.pkl','wb') as outfile:
		pkl.dump(tokens,outfile)
	with open('analysis/score.pkl','wb') as outfile:
		pkl.dump(score,outfile)

print('>>> saving token values')
with open('analysis/tokens.pkl','wb') as outfile:
	pkl.dump(tokens,outfile)

print('>>> saving calculated scores')
with open('analysis/score.pkl','wb') as outfile:
	pkl.dump(score,outfile)
