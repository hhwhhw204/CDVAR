import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from sklearn.metrics import roc_curve, auc, average_precision_score
sys.path.append("/home/hyz/workspace/hhw/DrugMutation/Reproduce/code/OncoKB_train/")
from model import CDVAR


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(42)

    for dataset in ["CIViC","CGI","JAX"]:
        data_path = f'../../data/eval/{dataset}'
        dna_fea = torch.load(f'{data_path}/dna_fea_except_oncokb.pt')
        prt_rep = torch.load(f'{data_path}/prt_rep_except_oncokb.pt')
        panfea = torch.load(f'{data_path}/panfea_data_except_oncokb.pt')
        labels = torch.load(f'{data_path}/labels_except_oncokb.pt')

        bs = 4
        test_dataset = TensorDataset(dna_fea, prt_rep, panfea, labels)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        model = CDVAR().cuda()
        model.load_state_dict(torch.load('../OncoKB_train/save_model/Pan-cancer_best_model_10.pth'))
        model.eval()

        target_total, output_total = [], []
        
        for dna_fea, prt_rep, panfea, target in test_loader:
            output = model(dna_fea.cuda(), prt_rep.cuda(), panfea.cuda())
            output = output.cpu().detach().numpy().flatten()
            target_total = np.concatenate((target_total, target))
            output_total = np.concatenate((output_total, output))

        fpr, tpr, _ = roc_curve(target_total, output_total)
        auroc = auc(fpr, tpr)
        auprc = average_precision_score(target_total, output_total)

        x1 = output_total[target_total == 0]
        x2 = output_total[target_total == 1]
        tatistic, pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative='two-sided') 
        print("dataset=",dataset,"auroc=",auroc,"auprc=",auprc,"p-value=",pvalue)