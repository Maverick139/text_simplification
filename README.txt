
Advanced NLP Project
Sentence Simplification using Wikipedia
=======================================================

Greetings!

Thank you for joining us in exploring the seemingly "simple" ;) problem of text-simplification
we found is increasingly complex and thought provoking. We have implemeted two approches:

1. Encoder-Decode Approach ( directory: enc_dec/ )
2. Statistical Machine Translation Approach ( directory: statMT/ )

Further instructions have been provided in the respective README.txt files of each approach.

NOTE: The current structure only contains the code files (due to size constraint). At this stage, the Enc-Dec approach is completely run-ready. For making the StatMT directory run-ready as well, download the (full) form of the project directory from the link given below:

https://drive.google.com/drive/folders/14XHBWoP3m8p6VwyoDrsjHRufTUYsTqWm?usp=sharing

The (full) directory structure of this project is as follows:

.
├── README.txt
├── data
│   └── wikiLarge
│       ├── test
│       │   ├── wiki.test.complex
│       │   └── wiki.test.simple
│       ├── train
│       │   ├── wiki.train.complex
│       │   └── wiki.train.simple
│       └── valid
│           ├── wiki.valid.complex
│           └── wiki.valid.simple
|
|
├── enc_dec
│   ├── analysis
│   │   ├── metadata.pkl
│   │   ├── record.pkl
│   │   ├── score.pkl
│   │   └── tokens.pkl
│   │   └── src
│   │       ├── create_controlled_eval_set.py
│   │       ├── custom_similarity_metric.py
│   │       ├── FastTextWordRank.pkl
│   │       ├── feature_extract.py
│   │       ├── parse_fairseq_generate.py
│   │       ├── req.py
│   │       ├── run_evaluation_metrics.py
│   │       └── to_dataframe.py
│   ├── data
│   │   ├── aug_data
│   │   ├── preproc_data
│   │   └── raw_data
│   │       ├── test
│   │       │   ├── wiki.test.complex
│   │       │   └── wiki.test.simple
│   │       ├── train
│   │       │   ├── wiki.train.complex
│   │       │   └── wiki.train.simple
│   │       └── valid
│   │           ├── wiki.valid.complex
│   │           └── wiki.valid.simple
│   ├── evaluate_demo.sh
│   ├── evaluate.sh
│   ├── models
│   ├── README.txt
│   ├── requirements.sh
│   ├── src
│   │   ├── create_controlled_eval_set.py
│   │   ├── custom_similarity_metric.py
│   │   ├── FastTextWordRank.pkl
│   │   ├── feature_extract.py
│   │   ├── parse_fairseq_generate.py
│   │   ├── __pycache__
│   │   │   └── custom_similarity_metric.cpython-36.pyc
│   │   ├── req.py
│   │   ├── run_evaluation_metrics.py
│   │   └── to_dataframe.py
│   ├── train_demo.sh
│   └── train.sh
|
|
├── statMT
│   ├── cmph-2.0
│   ├── corpus
│   ├── giza-pp
│   ├── lm
│   ├── models
│   │   ├── StatMT_137k.translate
│   │   └── StatMT_237k.translate
│   ├── mosesdec2
│   ├── README.txt
│   └── workspace
