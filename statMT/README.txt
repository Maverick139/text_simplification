Advanced NLP Project
Sentence Simplification using Wikipedia
Statistical Machine Translation Approach

This section of the project contains the sentence simplification model trained using statistical machine translation. 
It is trained on the Moses decoder released at http://www.statmt.org/moses/

The model made available is trained on the Simple English Wikipedia provided by Coster and Kauchak, as part of their paper Simple English Wikipedia - A New Text Simplification Task (2011) {paper available at https://www.aclweb.org/anthology/P11-2117/}

The raw dataset is provided in EXP/corpus/raw, split into train, tune and test sets as provided by the authors. A true-cased, cleaned version that is used for the training has been provided in EXP/corpus/, and is the default dataset used for training the models.

INSTRUCTIONS TO SETUP

The EXP directory already contains compiled binaries of both the Moses system, as well as GIZA++, mkcls, and cmph libraries that are required to train a model using Moses. However, some paths need to be configured before running the provided model on a sample input.

1. In the file EXP/workspace/binary/moses.ini, update the 'path' parameter of each of the features to point to the absolute path to the mentioned files. Specifically, the part of the path before the EXP dir will need to be updated based on the location where this folder is present on your machine.

2. Similar changes must be made to the files EXP/workspace/mert-work/moses.ini, as well as EXP/workspace/train/model/moses.ini

3. If the paths are correctly updated, you can try to obtain a sample output by using the following command from the workspace directory:
	
	echo "This is the sample input sentence. " | /PATH/TO/EXP/mosesdec2/bin/moses -f PATH/TO/workspace/binary/moses.ini

4. For checking outputs of multiple sentences, input can be provided through file, and output also dumped in a file. The inpu file must contain each sentence in a new line.

	/PATH/TO/EXP/mosesdec2/bin/moses -f PATH/TO/workspace/binary/moses.ini < /PATH/TO/input_file > /PATH/TO/output_file

5. To obtain BLEU score on the test file provided in the corpus, simply run the eval.sh script file in bash. To obtain it on other input and out, replace the "test.clean.simp" and "test.clean.comp" files inthe EXP/corpus/ directory with the reference output and the input respectively. Note that the file names of the new files must be the same as the default files, i.e. "test.clean.comp" and "test.clean.simp".

6. To train a model on the provided corpus, simply run EXP/workspace/train_model.sh, followed by EXP/workspace/binarise.sh from the workspace directory.

7. To train a model on a different corpus, split the corpus in sentence aligned train, test and tune sets, and replace the files in EXP/corpus/raw directory. As before, ensure the new files are named the same as the old files. After this, run the scripts clean_corpus.sh, train_model.sh, and binarise.sh from the workspace directory in that order. To evaluate the model, run eval.sh from the same directory as well.

IN CASE OF MOSES FAILURE

Though compiled binaries are provided, it is possible that moses may not be able to run correctly and throw errors during run time. In this case, moses will need to be recompiled to work correctly. To avoid irregularities, it would be advisable to recompile cmph as well.

This can be done by following these steps:


1. (Optional, do this if error persists) Remove the cmph-2.0 directory from EXP, and download the latest version from https://sourceforge.net/projects/cmph/
2. (Optional)Compile cmph according to the instructions provided in the library.
3. Ensure the following packages are installed, use sudo apt-get install <package> to install any missing ones:
   
   git
   subversion
   make
   libtool
   gcc
   g++
   libboost-dev
   tcl-dev
   tk-dev
   zlib1g-dev
   libbz2-dev
   python-dev
   libicu-dev (Debian)
   libunistring-dev (Debian)

4. Navigate to the mosesdec2 directory
5. Remove the bin directory.
6. Recompile moses with the following instruction:
	
	./bjam --with-cmph=/PATH/TO/cmph-2.0 -j4
	
	(note - -jX implies the compilation occurs in X threads, replace with number of CPUs available for fastest time.)

7. The process will take time depending on hardware. Once the compilation is successful, you should be able to train or test models according to the instructions provided above.

NOTE: Moses installation is tricky and can run into a number of problems. The steps provided were the steps followed for compilation on my local machine. Additional issues can occur to due to other pre-requisites such as boost, GIZA++, or mkcls. For full installation instructions, please refer the moses manual avaliable at: http://www.statmt.org/moses/manual/manual.pdf
