![logo](logo.png)
# CDVAR: Prioritization of Cancer Druggable Mutations Using Protein Structural Modeling

## :bulb:Introduction

CDVAR, a mutation-level target prediction model that prioritizes dynamic modeling of protein structural changes and integrates DNA, protein structural, and mutation modalities into a unified framework. 

![CDVAR](CDVAR.png)

## :bookmark_tabs: Model and Data

The optimal model from cross-validation, along with the training and evaluation data, is saved in [Google Drive](https://drive.google.com/drive/folders/1pK3Eey6F1t6uTw9JdQeHsNqMtF1486aq?usp=sharing). Downloading these files is essential for reproducing our results.

If the DNABERT2 model fails to load, please download it from [Huggingface](https://huggingface.co/zhihan1996/DNABERT-2-117M/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1pK3Eey6F1t6uTw9JdQeHsNqMtF1486aq?usp=sharing). 

The folder structure is as follows:

```
CDVAR/
├── CDVAR_score/
├── code/
│   ├── benchmark_predict/
│   ├── OncoKB_train/
│   │   └── save_model/
│   ├── processing/
│   │   ├── DNABERT_2_117M/
│   │   └── hub/
└── data/
    ├── asseq/
    │   ├── dna_asseq/
    │   └── prt_asseq/
    ├── eval/
    │   ├── CGI/
    │   ├── CIViC/
    │   └── JAX/
    ├── feature/
    │   ├── dna_repr/
    │   ├── prt_repr/
    │   └── statistics/
    ├── input/
    └── utils/
```



## :wrench: Setup environment

Clone this repository and cd into it as below.

```
git clone https://github.com/hhwhhw204/CDVAR.git
cd CDVAR
```

Prepare the environment,the `requirements.txt` is based on python3.8.0.

```
# create and activate virtual python environment
conda create --name cdvar python=3.8.0
conda activate cdvar
pip install -r requirements.txt
```

## 🧬 Usage

We have preprocessed the features required for training and stored them in `CDVAR/data/feature`. 

If you want to obtain multimodal features for your own mutation data, please follow the Data Processing steps. Otherwise, you can skip to the training step.

### 1. Data processing

Please prepare a CSV file with chromosome, position, reference base, and alternate base, following the format in  `CDVAR/data/input/example.csv `.

#### Obtain the initial sequence

We use the [hg19 genome](https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz) to obtain the DNA sequence, similar to the `CDVAR/data/asseq/dna_asseq/` format. 

```
# obtain the truncated DNA sequence.
# Please ensure that the file `CDVAR/data/utils/hg19.fa` exists.
cd code/processing/
python obtain_dna_sequence.py --dataset example
```

Use the [ANNOVAR](http://www.openbioinformatics.org/annovar/download/0wgxR2rIVP/annovar.latest.tar.gz)  to obtain protein annotations, similar to the `CDVAR/data/asseq/prt_asseq/` format. 

```
# An example of ANNOVAR usage is as follows
annotate_variation.pl input.tsv humandb/ -buildver hg19 -out output -exonsort
coding_change.pl output.exonic_variant_function humandb/hg19_refGene.txt humandb/hg19_refGeneMrna.fa -includesnp -out output_asseq
```

After that, run 

```
# obtain the truncated Protein sequence.
python obtain_prt_sequence.py  --dataset example
```

#### Obtain the representation

```
# The DNA sequence is processed with DNABERT2 to extract final-layer representations.
python obtain_dna_repr.py --dataset example
# The DNA sequence is processed with ESM2 to extract final-layer representations.
python obtain_prt_repr.py --dataset example
# Obtain mutation statistic features.
python obtain_cancer_mutation_multipool.py --dataset example
```

### 2. Train

We sampled negative examples from COSMIC and, due to the database's large size, only provide their multimodal features for training.

```
cd code/OncoKB_train
python pan-caner_train.py --mul 10 --e 8 --bs 12 --lr 0.00005
python specific-caner_train.py --cancer AML --e 8 --bs 12 --lr 0.00005
```

The optimal training model is saved in `CDVAR/code/OncoKB_train/save_model/`.

### 3. Benchmark Predict

The evaluation data is in `CDVAR/data/eval`. Due to storage limitations, we have only performed prediction on the CIViC and CGI datasets to facilitate result reproduction in the paper.

```
cd code/benchmark_predict
python pan-cancer_predict.py
```

### 4. CDVAR score
CDVAR_score/ directory are stored the CDVAR scores for 267,679 mutations from the COSMIC Census Genes Mutation project and 3,289,953 mutations from the COSMIC Cancer Mutation Census. 

The prediction results can also be viewed on the website http://cdvar.haiyanglab.cn/.

