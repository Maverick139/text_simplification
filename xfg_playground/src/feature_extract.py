import numpy as np
import Levenshtein
import spacy
import pickle as pkl
from nltk.corpus import stopwords
from os import path
import argparse
from tqdm import tqdm

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

# consolidated run function
###########################################################################################

def modify(res):
    '''
        lower all text, prepend relational values
    '''
    for phase in ['test','valid','train']:
        # input files of the form: '{dirPath}/wiki.{phase}.complex'
        data = {'complex': list(),
                'simple': list()}

        inp_file = {'complex': path.join(res.source_dir,phase,'wiki.{}.complex'.format(phase)), 
                    'simple': path.join(res.source_dir,phase,'wiki.{}.simple'.format(phase))}

        # read complex file
        for typ in ['complex','simple']:
            with open(inp_file[typ]) as infile:
                for line in infile.readlines():
                    line = line.lower()
                    data[typ].append(line.lower().rstrip('\n'))

        samples = len(data['complex']) if len(data['complex'])==len(data['simple']) else None
        if not samples:
            print('>>> ERROR: comp/simp file size mismatch') 

        complexOutData, simpleOutData = list(), list()

        relation_data = list()

        print('>>> augmenting {}'.format(path.join(res.source_dir,phase)))
        print('>>> tokens: NbChars:{}, LevSim:{}, WordRank:{}, DepTreeDepth:{}'.format(res.nbchars,res.levsim,res.wrdrank,res.deptree))
        for i in tqdm(range(samples)):

            comp, simp = data['complex'][i], data['simple'][i]

            relStrings = list()

            #1. 
            if res.nbchars==True:
                charRatio = characterRatio(comp.split(),simp.split())
                relStrings.append('<NbChars_{}>'.format(round(charRatio,2)))
            #2.
            if res.levsim==True:
                levRatio = levenshtein_similarity(comp, simp)
                relStrings.append('<LevSim_{}>'.format(round(levRatio,2)))
            #3. 
            if res.wrdrank==True:
                rankComp, rankSimp = lexical_complexity(comp.split()), lexical_complexity(simp.split())
                rankRatio = rankSimp/rankComp if rankComp>0 else 0
                relStrings.append('<WordRank_{}>'.format(round(rankRatio,2)))
            #4.
            if res.deptree==True:
                depComp, depSimp = dependency_tree_depth(comp), dependency_tree_depth(simp)
                depRatio = depSimp/depComp if depComp>0 else 0
                relStrings.append('<DepTreeDepth_{}>'.format(round(depRatio,2)))

            #relStrings = ['<NbChars_{}>'.format(round(charRatio,2)),
            #                '<LevSim_{}>'.format(round(levRatio,2)),
            #                '<WordRank_{}>'.format(round(rankRatio,2)),
            #                '<DepTreeDepth_{}>'.format(round(depRatio,2))]
            
            augComp = relStrings if len(relStrings)!=0 else list()
            augComp.extend([comp])
            augCompString = ' '.join(augComp)
            complexOutData.append(augCompString+'\n')
            simpleOutData.append(simp+'\n')

            # print(augCompString)
            # print(simp)
            # print('---------------------------------')
            # _ = input()

        with open(path.join(res.dest_dir,phase,'wiki.{}.complex'.format(phase)),'w') as outfile:
            outfile.writelines(complexOutData)

        with open(path.join(res.dest_dir,phase,'wiki.{}.simple'.format(phase)),'w') as outfile:
            outfile.writelines(simpleOutData)

        print('>>> completed. saved as {}'.format(path.join(res.dest_dir,phase)))

###########################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--source', action='store', dest='source_dir', type=str, default='raw_data',
                        help='source folder (contains train test valid)')
    parser.add_argument('-d','--dest', action='store', dest='dest_dir', type=str, default='aug_data',
                        help='destination folder (contains train test valid)')
    parser.add_argument('--nbchars', action='store_true', dest='nbchars', default='False',
                        help='prepend NbChars token')
    parser.add_argument('--levsim', action='store_true', dest='levsim', default='False',
                        help='prepend LevSim token')
    parser.add_argument('--wrdrank', action='store_true', dest='wrdrank', default='False',
                        help='prepend WordRank token')
    parser.add_argument('--deptree', action='store_true', dest='deptree', default='False',
                        help='prepend DepTreeDepth token')    
    res = parser.parse_args()

    stopwords = stopwords.words('english')
    PARSER = init_spacy_model('en_core_web_sm')
    logRank = buildRank('src/FastTextWordRank.pkl')
    modify(res)
