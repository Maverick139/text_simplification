Advanced NLP Project
Sentence Simplification using Wikipedia
Encoder-Decoder Translation Approach

NOTE: This directory contains two implemetations of the En-Dec approach: 

	A. FairSeq Implemetation (primary)
	B. From-Scratch Implementation

Setup instructions for the primary implemetation (using Facebook's FairSeq) have been given below. For instructions for our from-sratch implementation of the model, scroll to the bottom of this document. 

INSTRUCTIONS FOR SETUP

# FairSeq Implemetation
---------------------------

The following steps will guide you through installing the basic libraries and dependencies required to run the enc-dec model. 

1. To simplify the setup process, you have been provided with an installation script: requirements.sh in the main directory. 

	>>> bash requirements.sh

	Run the same to install the following dependencies:
	
	a. FairSeq tool-kit
	b. SpaCy en_core_web_sm language model
	c. Levenshtein similarity module
	d. NLTK Stopwords

2. Evaluating model checkpoints

	# Note that checkpoint evaualtion also requires the above mentioned dependencies

	a. Download checkpoint_demo.zip and extract the directory

	link: https://drive.google.com/drive/folders/1dVKXIjvzKsSsOA5Ad1kczrCOEjTvfB1C?usp=sharing

	b. Move into the checkpoint_demo directory
	c. Run the following commands to generate outfile sentences:

		>>> rm -rf demo_eval; mkdir demo_eval

		>>> fairseq-generate preproc_data --path checkpoint15.pt --batch-size 64 --beam 5 --remove-bpe > demo_eval/fsq_gen_test_1110_15.txt

	d. Followed by the command below, to get the SARI4 score of the same:

		>>> python3 src/parse_fairseq_generate.py -g demo_eval --sari

3. Training on local machine
	
	For low-compute machines, we have provided demo training and evaluation scripts for a highly-sized down model. 

		>>> bash train_demo.sh 		# To be terminated manually after 5 epochs
		>>> bash evaluate_demo.sh 	# will print model output on WikiLarge Test set (359 sentences)
						# S: source (complex) sentence
						# T: target (simple) sentence
						# H: predicted simplified model output

If you still wish to run the full-scale model to obtain the results mentioned in the paper, refer to the following instructions:

4. Training on ADA

	a. Copy the entire parent directory to the /scratch folder of any gnode(01-61)
	b. Start live/remote session with CPUs: 35 (min) & GPUs: 4 (min)
	c. Run the train script:

		>>> bash train.sh

	Note: You will have to manually terminate the script when the desired number of epochs have been completed

	d. Run the evaluation script:

		>>> bash evaluate.sh

	e. Finally, run the metric performance script:

		>>> python3 src/parse_fairseq_generate.py -g checkPoints --sari

	to view the SARI4 scores of each of the epoch checkpoints


# From-Scratch Implemetation
---------------------------





