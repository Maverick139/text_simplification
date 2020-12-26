import numpy as np
from nltk import ngrams

def f_score(p,r):
	if not p or not r:
		return 0
	return 2*p*r/(p+r)

def _sari_add(ngrams_list):

	s_prec = list()
	s_rec = list()

	for n in range(len(ngrams_list)):
		n_score = 0
		inp, ref, out = ngrams_list[n]
		for gram in out:
			n_score += min(gram in out and gram not in inp, gram in ref)

		den1 = sum([gram in out and gram not in inp for gram in out])
		den2 = sum([gram in ref and gram not in inp for gram in ref])

		prec = n_score/den1 if den1!=0 else 1
		rec = n_score/den2 if den2!=0 else 1

		s_prec.append(prec)
		s_rec.append(rec)

	# print(s_prec,s_rec)
	return np.mean(s_prec), np.mean(s_rec) 

def _sari_keep(ngrams_list):

	s_prec = list()
	s_rec = list()

	for n in range(len(ngrams_list)):
		n_score = 0
		inp, ref, out = ngrams_list[n]
		for gram in inp:
			n_score += min(gram in inp and gram in out, gram in inp and gram in ref)

		den1 = sum([gram in inp and gram in out for gram in out])
		den2 = sum([gram in inp and gram in ref for gram in out])

		prec = n_score/den1 if den1!=0 else 1
		rec = n_score/den2 if den2!=0 else 1

		s_prec.append(prec)
		s_rec.append(rec)

	# print(s_prec,s_rec)
	return np.mean(s_prec), np.mean(s_rec)

def _sari_del(ngrams_list):

	s_prec = list()
	s_rec = list()

	for n in range(len(ngrams_list)):
		n_score = 0
		inp, ref, out = ngrams_list[n]
		for gram in inp:
			n_score += min(gram in inp and gram not in out, gram in inp and gram not in ref)

		den1 = sum([gram in inp and gram not in out for gram in inp])
		den2 = sum([gram in inp and gram in ref for gram in out])

		prec = n_score/den1 if den1!=0 else 1
		rec = n_score/den2 if den2!=0 else 1

		s_prec.append(prec)
		s_rec.append(rec)

	# print(s_prec)
	return np.mean(s_prec), np.mean(s_rec)

def SARI4(inp,ref,out,n=4,d=[1/3,1/3,1/3]):

	# inp, ref, out = inp.lower(), ref.lower(), out.lower()
	# inp, ref, out = inp.split(), ref.split(), out.split()

	min_len = min(len(inp),len(ref),len(out))
	n_fix = min(n,min_len)
	ngrams_list = [[] for i in range(n_fix)]

	for i in range(n_fix):
		ngrams_list[i] = [list(ngrams(inp,i)),list(ngrams(ref,i)),list(ngrams(out,i))] 	

	P_add, R_add = _sari_add(ngrams_list)
	P_keep, R_keep = _sari_keep(ngrams_list)
	P_del, R_del = _sari_del(ngrams_list)

	# print(P_add,R_add)
	# print(P_keep,R_keep)
	# print(P_del)

	F_add = f_score(P_add,R_add)
	F_keep = f_score(P_keep,R_keep)

	sari = d[0]*F_add + d[1]*F_keep + d[2]*P_del

	return sari, {	'add':{'prec':P_add,'rec':R_add},
					'keep':{'prec':P_keep,'rec':R_keep},
					'del':{'prec':P_del,'rec':R_del}}


# inp = "About 95 species are currently accepted ."
# ref = "About 95 species are currently known ."
# out = "About 95 species are currently agreed ."

# print(SARI4(inp,ref,out))
