![logo](logo.png)
### CDVAR: A Multimodal Model for Precise Prediction of Druggable Mutations in Cancer

### 1.Introduction

CDVAR, a mutation-level target prediction model that prioritizes dynamic modeling of protein structural changes and integrates DNA, protein structural, and mutation modalities into a unified framework. 

![CDVAR](CDVAR.png)

### 2. Model and Data

The optimal model from cross-validation, along with the training and evaluation data, is saved in xxx. Downloading these files is essential for reproducing our results.

If the DNABERT2 model fails to load, please download it from https://huggingface.co/zhihan1996/DNABERT-2-117M/tree/main. If the ESMfold model fails to load, please download it from https://github.com/facebookresearch/esm/tree/main.

### 3. Setup environment

```
git clone https://github.com/hhwhhw204/CDVAR.git
# create and activate virtual python environment
cd CDVAR
conda env create -f environment.yml
```

### 4.Usage

### Data processing

Please prepare a CSV file with chromosome, position, reference base, and alternate base, following the format in  `CDVAR/data/input/example.csv `.

#### Obtain the initial sequence.

We use the hg19 genome (https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz) to obtain the DNA sequence based on the mutation site. Please ensure that the file `CDVAR/data/utils/hg19.fa` exists.

```
# obtain the truncated DNA sequence.
python obtain_dna_sequence.py -dataset example
```

Use the ANNOVAR(http://www.openbioinformatics.org/annovar/download/0wgxR2rIVP/annovar.latest.tar.gz)  to obtain protein annotations, similar to the `CDVAR/data/example/example_asseq` format. 

```
# An example of ANNOVAR usage is as follows
annotate_variation.pl input.avinput humandb/ -buildver hg19 -out output -exonsort
coding_change.pl output.exonic_variant_function humandb/hg19_refGene.txt humandb/hg19_refGeneMrna.fa -includesnp -out output_asseq
```

After that, run 

```
# obtain the truncated Protein sequence.
python obtain_prt_sequence.py  -dataset example
```

#### Obtain the representation

```
# The DNA sequence is processed with DNABERT2 to extract final-layer representations.
python obtain_dna_repr.py -dataset example
# The DNA sequence is processed with ESM2 to extract final-layer representations.
python obtain_prt_repr.py -dataset example
# Obtain mutation statistic features.
python obtain_cancer_mutation_multipool.py -dataset example
```



### Train

We sampled negative examples from COSMIC and, due to the database's large size, only provide their multimodal features for training.

```
cd code/OncoKB_train
python pan-caner_train.py -mul 10 -e 8 -bs 12 -lr 0.00005
python specific-caner_train.py -cancer AML -e 8 -bs 12 -lr 0.00005
```

The optimal training model is saved in `CDVAR/code/OncoKB_train/save_model/`.



### Benchmark Predict

We have preprocessed the prediction data from the CIViC, CGI, and JAX-CKB datasets to facilitate result reproduction in paper. The evaluation data is in `CDVAR/data/eval`.

```
cd code/benchmark_predict
python pan-cancer_predict.py
```



### CDVAR score
CDVAR_score/ directory are stored the CDVAR scores for 267,679 mutations from the COSMIC Census Genes Mutation project and 3,289,953 mutations from the COSMIC Cancer Mutation Census. 

The prediction results can also be viewed on the website http://cdvar.haiyanglab.cn/.

