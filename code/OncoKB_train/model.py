import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import math
from utils import *
from rtdl_num_embeddings import PeriodicEmbeddings


class PeriodicEmb(nn.Module):
    def __init__(self, num_cont=81, dna_size=768, lite=True):
        super().__init__()
        self.emb = PeriodicEmbeddings(num_cont, dna_size, lite=lite)
    def forward(self, pfea):
        pfea = self.emb(pfea)
        return pfea
    

class MLP(nn.Module):
    def __init__(self,dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(81),
            nn.Dropout(0.2),
            nn.Linear(dim // 4, dim),
        )
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

   

class MHAtt(nn.Module):
    def __init__(self, hidden_dim, head, dropout_r):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / head)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim,hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape
        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)
        atted,att_map = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)
        return atted,att_map

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map0 = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map0)
        return torch.matmul(att_map, value),att_map0
    


class CrossAttn(nn.Module):
    def __init__(self,size,num_heads=4,dropout=0.5, CA_ffn=False):
        super(CrossAttn, self).__init__()
        self.CA_ffn = CA_ffn
        self.enc = MHAtt(size, num_heads, dropout)
        self.ln = nn.LayerNorm(size)
        self.drop = nn.Dropout(0.2)
        if CA_ffn:
            self.ffn = FFN(size)
            
    def forward(self, x, y):
        x_out,attn_map = self.enc(y, y, x) 
        x_out = self.drop(x_out)
        x = self.ln(x_out + x)
        if self.CA_ffn:
            x = self.ffn(x)
        return x, attn_map
    


class FFN(nn.Module):
    def __init__(self,size):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
                            nn.Linear(size, size*2),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(size*2, size)
                        )
        self.ln_ffn = nn.LayerNorm(size)
    def forward(self,x):
        return self.ln_ffn(x + self.ffn(x))
    

  

class FCA(nn.Module):
    def __init__(self,size,num_heads=4,dropout=0.5, CA_ffn=False):
        super(FCA, self).__init__()
        self.CA_ffn = CA_ffn
        self.enc = MHAtt(size, num_heads, dropout)
        self.ln = nn.LayerNorm(size)
        self.drop = nn.Dropout(0.2)
        if CA_ffn:
            self.ffn = FFN(size)
            
    def forward(self, x, y):
        K = torch.cat((x,y), dim=1)     
        V = torch.cat((x,y), dim=1)
        x_out,attn_map = self.enc(V, K, x) 
        x_out = self.drop(x_out)
        x = self.ln(x_out + x)
        if self.CA_ffn:
            x = self.ffn(x)
        return x,attn_map
    



class CDVAR(nn.Module):
    def __init__(self, num_heads=4, dropout=0.5, dna_size=768, prt_size=480, num_cont=81, CA_ffn=False,FCA_ffn=False):
        super(CDVAR, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=dna_size, out_channels=dna_size, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=prt_size, out_channels=dna_size, kernel_size=1, stride=1, padding=0)
        self.emb = PeriodicEmbeddings(num_cont, 768, lite=True)
        self.mlp = MLP(dna_size)

        self.CA1_1 = CrossAttn(dna_size, num_heads, dropout,CA_ffn)
        self.CA1_2 = CrossAttn(dna_size, num_heads, dropout,CA_ffn)
        self.CA2_1 = CrossAttn(dna_size, num_heads, dropout,CA_ffn)
        self.CA2_2 = CrossAttn(dna_size, num_heads, dropout,CA_ffn)
        self.FCA = FCA(dna_size, num_heads, dropout, FCA_ffn)

        self.alpha = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flt = nn.Flatten(1)
        self.linear1 = nn.Linear(dna_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dna, prt, pfea):
        # Feature Projection 
        dna = self.conv1(dna.transpose(1,2)).transpose(1,2)
        prt = self.conv2(prt.transpose(1,2)).transpose(1,2)
        pfea = self.emb(pfea)
        pfea = self.mlp(pfea)

        # DNA-Protein Module
        dna_prt1, _ = self.CA1_1(dna, prt)  
        dna_pfea1, _ = self.CA1_2(dna, pfea)  

        # DNA-Mutation Module
        dna_prt2, _ = self.CA2_1(dna_prt1, prt)
        dna_pfea2, _ = self.CA2_2(dna_pfea1, pfea)
        
        # Adaptive Summation
        dna_prt = self.alpha * dna_prt1 + (1 - self.alpha) * dna_prt2
        dna_pfea = self.beta * dna_pfea1 + (1 - self.beta) * dna_pfea2
        concat_fea = torch.cat([dna_prt,dna_pfea],dim=1)

        # Deep Fusion Module
        dna_prt_pfea,_ = self.FCA(dna, concat_fea)

        dna_avg_out = self.avgpool(dna_prt_pfea.transpose(1, 2))            
        dna_flt_out = self.flt(dna_avg_out)
        out = self.linear1(dna_flt_out)
        out = self.sigmoid(out)
        return out