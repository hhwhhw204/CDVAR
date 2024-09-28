![logo](logo.png)
### CDVAR: A Multimodal Model for Precise Prediction of Druggable Mutations in Cancer

CDVAR, a groundbreaking multimodal model that predicts druggable mutations with unparalleled accuracy across both pan-cancer and specific cancer types. By seamlessly integrating a diverse spectrum of features, including DNA and protein sequences, and mutation annotations, CDVAR constructs a powerful framework for mutation prediction. 

![CDVAR](CDVAR.png)

### Data processing

The code for processing data features is located in the processing/ directory. Please prepare a CSV file containing the chromosome, mutation position, reference base, and alternate base, following the format of info/example.csv. Use the ANNOVAR tool to obtain protein annotations, similar to the info/example_asseq format. After that, run 
```
python obtain_prt_sequence.py
```
obtain the truncated protein sequence.

```
python obtain_prt_repr.py
```
obtain the truncated protein representation.
 
```
python obtain_dna_sequence.py
```
obtain the truncated DNA sequence.

```
python obtain_dna_repr.py
```
obtain the truncated DNA representation.

```
python obtain_cancer_mutation_multipool.py
```
obtain cancer context features using multiprocessing.

The features required for training are the truncated protein representation, truncated DNA sequence, and cancer features

### Train

The training code is in the OncoKB_train/ directory. DNA sequence was passed through DNABERT2 pre-training model, protein sequence was passed through ESMFold model, and middle layer characterization was extracted as model data. After data processing, run

```
python pan-caner_train.py -mul 10
```

### Benchmark Predict

The prediction codes for the other baseline datasets are located in benchmark_predict/. Again, after the data is processed, run

```
python pan-cancer_predict.py
python specific-cancer_predict.py
```

### CDVAR score
Here are stored the CDVAR scores for 267,679 mutations from the COSMIC Census Genes Mutation project and 3,289,953 mutations from the COSMIC Cancer Mutation Census.


