import random
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from model import CDVAR
from utils import *
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


        

def k_verify(X, P, F, Y, args):
    X, P, F, Y = shuffle(X, P, F, Y, random_state=0)

    k = 5
    n = Y.shape[0]
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))[:n]

    rocs,accs,auprcs = [],[],[]
    x1,x2 = [],[]
    min_val_loss = float('inf')

    for i in range(1, k + 1):
        torch.cuda.empty_cache()

        net = CDVAR().to(device)
        criterion = FocalLoss(alpha=0.75, gamma=2)
        optimizer = optim.Adam(params=net.parameters(),
                               lr=args.lr, betas=(0.9, 0.999))

        ix = assignments == i
        train_loader = DataLoader(TensorDataset(torch.tensor(X[~ix]), torch.tensor(P[~ix]), torch.tensor(F[~ix]), torch.tensor(Y[~ix])),
                                  batch_size=args.bs, shuffle=False, drop_last=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X[ix]), torch.tensor(P[ix]), torch.tensor(F[ix]), torch.tensor(Y[ix])),
                                 batch_size=48, shuffle=False)
        
        target_total, output_total, val_losses = torch.tensor([]), torch.tensor([]), []
        for epoch in range(args.e):
            for dna, prt, panfea, target in train_loader:
                net.train()
                dna, prt, panfea, target = dna.to(device), prt.to(device), panfea.to(device), target.to(device)
                output = net(dna,prt,panfea).flatten()
                loss = criterion(output.float(), target.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        for dna, prt, panfea, target in test_loader:
            dna, prt, panfea, target = dna.to(device), prt.to(device), panfea.to(device), target.to(device)
            with torch.no_grad():
                output = net(dna,prt,panfea).flatten()
                val_loss = criterion(output.float(), target.float()).cpu()
                val_losses.append(val_loss.detach().numpy())
                val_right = rightness(output, target)
                target_total = torch.cat((target_total, target.cpu()))
                output_total = torch.cat((output_total, output.cpu()))

        if sum(val_losses) < min_val_loss:
            min_val_loss = sum(val_losses)
            torch.save(net.state_dict(), f"save_model/{args.cancer}_best_model_tmp.pth")
        print(f'Fold{i} Validation loss: {val_loss:.4f}')
        accs.append(100. * val_right)


        fpr, tpr, _ = roc_curve(target_total, output_total)
        rocs.append(auc(fpr, tpr))

        auprc = average_precision_score(target_total, output_total)
        auprcs.append(auprc)

        x1.extend(output_total[target_total == 0].numpy())
        x2.extend(output_total[target_total == 1].numpy())

    mean_auroc = np.mean(rocs, axis=0)
    print("Mean AUROC (area = %0.4f)" % mean_auroc)

    mean_auprc = np.mean(auprcs, axis=0)
    print("Mean AUPRC (area = %0.4f)" % mean_auprc)

    mean_acc = np.mean(accs, axis=0)
    print("Mean acc  = %0.4f" % mean_acc)

    tatistic, pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative='two-sided') 
    print("pvalue= %4e" % pvalue)
    return mean_auroc, mean_auprc, pvalue, mean_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--cancer', type=str, default='NSCLC',choices=['AML','BRCA','NSCLC'])
    parser.add_argument('--e', type=int, default=10)
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()
    set_seed(42) 

    neg_sample = f"COSMIC_sample4OncoKB_{args.cancer}_mul{args.mul}_seed42"
    print("Loading processed prt repr by Esmfold...")
    pos_prt_path = f"../../data/feature/prt_repr/OncoKB/OncoKB_{args.cancer}_prt_200aa_repr.npy" 
    neg_prt_path = f"../../data/feature/prt_repr/COSMIC/{neg_sample}_prt_200aa_repr.npy" 
    prt_rep, labels = concat_prt_repr(pos_prt_path, neg_prt_path)
    
    print("Loading processed dna repr by DNABERT2...")
    pos_dna_path = f"../../data/feature/dna_repr/OncoKB/OncoKB_{args.cancer}_dna_500bp_repr.npy" 
    neg_dna_path = f"../../data/feature/dna_repr/COSMIC/{neg_sample}_dna_500bp_repr.npy" 
    dna_fea = concat_dna_repr(pos_dna_path, neg_dna_path)

    print("Loading processed Mutation statistics...")
    pos_panfea_path = f"../../data/feature/statistics/OncoKB/OncoKB_{args.cancer}_100bpx2_statistics.tsv"
    neg_panfea_path = f"../../data/feature/statistics/COSMIC/{neg_sample}_100bpx2_statistics.tsv" 
    panfea = concat_pancan_fea(pos_panfea_path, neg_panfea_path)

    # k-folf cross verify
    print("Start Training...")
    auroc, auprc, pvalue, acc = k_verify(dna_fea, prt_rep, panfea, labels, args)
