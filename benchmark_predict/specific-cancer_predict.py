import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from sklearn.metrics import roc_curve, auc, average_precision_score

sys.path.append("/root/capsule/code/OncoKB_train/")
from model import CDVAR


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(42)

    cancer = "NSCLC"
    output_file_path = "../../results/specific-cancer_predict.txt"
    for dataset in [f"CIViC_{cancer}",f"CGI_{cancer}",f"JAX-CKB_{cancer}"]:
        data_path = f'../../data/{dataset}/'
        dna_fea = torch.load(f'{data_path}/{cancer}_dna_fea.pt')
        prt_rep = torch.load(f'{data_path}/{cancer}_prt_rep.pt')
        panfea = torch.load(f'{data_path}/{cancer}_panfea_data.pt')
        labels = torch.load(f'{data_path}/{cancer}_labels.pt')

        bs = 4
        test_dataset = TensorDataset(dna_fea, prt_rep, panfea, labels)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        model = CDVAR().cuda()
        model.load_state_dict(torch.load(f'../OncoKB_train/out/{cancer}_best_model.pth'))
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
        
        result_str = f"dataset={dataset}, auroc={auroc:.4f}, auprc={auprc:.4f}, p-value={pvalue:.6e}\n"
        with open(output_file_path, "a") as f:
            f.write(result_str)
        print(result_str)