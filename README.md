### CDVAR:

CDVAR, a groundbreaking multimodal model that predicts druggable mutations with unparalleled accuracy across both pan-cancer and specific cancer types. By seamlessly integrating a diverse spectrum of features, including DNA and protein sequences, and mutation annotations, CDVAR constructs a powerful framework for mutation prediction. 

![CDVAR](CDVAR.png)

### Data processing

The code for processing data characteristics is in the processing/ directory. Please prepare a csv containing the chromosome, mutation position, reference base and alternative base. The ANNOVAR tool and the code under processing/ can be used to obtain the reference sequence and alternative sequence of DNA after interception, the reference sequence and mutation alternative of protein after interception, and the statistical characteristics of cancer mutation context.

#### Train

The training code is in the OncoKB_train/ directory. DNA sequence was passed through DNABERT2 pre-training model, protein sequence was passed through ESMFold model, and middle layer characterization was extracted as model data. After data processing, run

```
python pan-caner_train.py -mul 10
```

#### Benchmark Predict

The prediction codes for the other baseline datasets are located in benchmark_predict/. Again, after the data is processed, run

```
python pan-cancer_predict.py
python specific-cancer_predict.py
```



