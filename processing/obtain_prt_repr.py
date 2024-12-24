import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import torch
import os
os.environ['TORCH_HOME']=''


def seq2rep(filename, model, alphabet):
    df = pd.read_csv(filename, sep='\t', header=None)
    elements = [("protein" + str(i), df.iloc[i, 0]) for i in range(len(df))]
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter(elements)

    batch_size = 15
    token_representations = []
    for i in tqdm(range(0, len(batch_tokens), batch_size)):
        batch_data = batch_tokens[i:i + batch_size].cuda()
        with torch.no_grad():
            results = model(batch_data, repr_layers=[12], return_contacts=False)
            batch_token_representations = results["representations"][12].cpu().detach().numpy()
        token_representations.append(batch_token_representations)

    # Concatenate all batches
    token_representations = np.concatenate(token_representations, axis=0)
    return token_representations


def obtain_prt_repr(ref_prt_asseq_file, alt_prt_asseq_file, save_prt_repr):
    # load model
    # The model reference: https://github.com/facebookresearch/esm
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.cuda().eval()

    ref_token_rep = seq2rep(ref_prt_asseq_file, model, alphabet)
    alt_token_rep = seq2rep(alt_prt_asseq_file, model, alphabet)
    token_reps = np.concatenate((ref_token_rep, alt_token_rep), axis=1)
    np.save(save_prt_repr, token_reps)   
    return token_reps




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument("--dataset", default='example', choices=['example','OncoKB','CIViC','CGI','JAX'])
    parser.add_argument("--cut_len", default=100)
    args = parser.parse_args()

    ref_prt_asseq =  f"../../data/asseq/prt_asseq/{args.dataset}/{args.dataset}_asseq_prt_ref_200aa.txt"
    alt_prt_asseq =  f"../../data/asseq/prt_asseq/{args.dataset}/{args.dataset}_asseq_prt_alt_200aa.txt"
    save_prt_repr = f"../../data/feature/prt_repr/{args.dataset}/{args.dataset}_prt_200aa_repr.npy"
    obtain_prt_repr(ref_prt_asseq, alt_prt_asseq, save_prt_repr)
