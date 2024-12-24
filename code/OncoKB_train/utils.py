import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


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



def rightness(predictions, labels):
    predictions = predictions.cpu()
    labels = labels.cpu()
    preds = (predictions > 0.5).float()
    acc = accuracy_score(labels, preds)
    return acc


 
class FocalLoss(nn.Module):
	def __init__(self,alpha=0.25,gamma=2):
		super(FocalLoss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self,preds,labels):
		eps=1e-7
		loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)