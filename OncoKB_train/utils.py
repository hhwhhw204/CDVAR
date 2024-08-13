import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


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