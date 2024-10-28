import random
import argparse
import warnings
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from scipy import stats
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


def concat_prt_repr(pos_path, neg_path):
    pos_prt_token = np.load(pos_path)
    neg_prt_token = np.load(neg_path)
    prt_repr = np.concatenate((pos_prt_token, neg_prt_token), axis=0)
    prt_repr = torch.tensor(prt_repr)
    labels = torch.tensor(np.array([1]*len(pos_prt_token) + [0]*len(neg_prt_token)))
    return prt_repr, labels


def concat_dna_repr(pos_path, neg_path):
    pos_dna_repr = np.load(pos_path)
    neg_dna_repr = np.load(neg_path)
    dna_repr = np.concatenate((pos_dna_repr,neg_dna_repr), axis=0)
    dna_repr = torch.tensor(dna_repr)
    return dna_repr


def concat_pancan_fea(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path,sep='\t')
    neg_df = pd.read_csv(neg_path,sep='\t')
    pan_df = pd.concat((pos_df,neg_df),axis=0)
    pan_df = pan_df.drop(['chr','pos','YES','.'],axis=1)
    panfea = torch.tensor(pan_df.values, dtype=torch.float32)
    return panfea
        

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
        target_total, output_total, val_losses = torch.tensor([]), torch.tensor([]), []

        net = CDVAR().to(device)
        criterion = FocalLoss(alpha=0.75, gamma=2)
        optimizer = optim.Adam(params=net.parameters(),
                               lr=args.lr, betas=(0.9, 0.999))

        ix = assignments == i
        train_loader = DataLoader(TensorDataset(torch.tensor(X[~ix]), torch.tensor(P[~ix]), torch.tensor(F[~ix]), torch.tensor(Y[~ix])),
                                  batch_size=args.bs, shuffle=False, drop_last=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X[ix]), torch.tensor(P[ix]), torch.tensor(F[ix]), torch.tensor(Y[ix])),
                                 batch_size=48, shuffle=False)

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
            torch.save(net.state_dict(), f"out/{args.cancer}_best_model.pth")
        print(f'Validation loss: {val_loss:.4f}  Accuracy: {100. * val_right:.2f}%')
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

    print("Loading prt repr by esmfold...")
    pos_prt_path = f"../processing/out/OncoKB/OncoKB_{args.cancer}_prt_200bp_repr.npy" 
    neg_prt_path = f"../processing/out/COSMIC/COSMIC_{args.cancer}_prt_200bp_repr.npy" 
    prt_rep, labels = concat_prt_repr(pos_prt_path, neg_prt_path)
    
    print("Loading dna repr by DNABERT2...")
    pos_dna_path = f"../processing/out/OncoKB/OncoKB_{args.cancer}_dna_500bp_repr.npy" 
    neg_dna_path = f"../processing/out/COSMIC/COSMIC_{args.cancer}_dna_500bp_repr.npy" 
    dna_fea = concat_dna_repr(pos_dna_path, neg_dna_path)

    print("Loading processed pancan fea...")
    pos_panfea_path = f"../processing/out/OncoKB/OncoKB_{args.cancer}_mut_fea_100bpx2.tsv"
    neg_panfea_path = f"../processing/out/COSMIC/COSMIC_{args.cancer}_mut_fea_100bpx2.tsv" 
    panfea,_ = concat_pancan_fea(pos_panfea_path, neg_panfea_path)

    ################ k-folf cross verify ###############
    auroc, auprc, pvalue, acc = k_verify(dna_fea, prt_rep, panfea, labels, args)
