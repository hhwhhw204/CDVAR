
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging
logging.set_verbosity_error()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_features(model, tokenizer, sequence):
    return [model(tokenizer(seq, return_tensors='pt')["input_ids"].cuda())[0].squeeze(0).cpu().detach().numpy() 
            for seq in tqdm(sequence)]


def obtain_dna_repr(ref_asseq_file, alt_asseq_file, save_dna_repr, args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("./DNABERT_2_117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("./DNABERT_2_117M", trust_remote_code=True).cuda()
    model.eval()
    
    # Load data
    ref_data = pd.read_csv(ref_asseq_file, header=None, sep='\t')[1]
    alt_data = pd.read_csv(alt_asseq_file, header=None, sep='\t')[1]
    ref_fea_list = process_features(model, tokenizer, ref_data)
    alt_fea_list = process_features(model, tokenizer, alt_data)

    # Padding
    max_len = args.seq_len // 2
    ref_padded_fea = np.array([np.pad(fea, ((0, max_len - fea.shape[0]), (0, 0)), 'constant') for fea in ref_fea_list])
    alt_padded_fea = np.array([np.pad(fea, ((0, max_len - fea.shape[0]), (0, 0)), 'constant') for fea in alt_fea_list])    
    
    # Combine and save
    dna_fea = np.concatenate((ref_padded_fea, alt_padded_fea), axis=1)
    np.save(save_dna_repr, dna_fea)
    return dna_fea




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument("--dataset", default='example', choices=['example','OncoKB','CIViC','CGI','JAX'])
    parser.add_argument("--seq_len", default=250)
    args = parser.parse_args()

    dna_ref_asseq = f"../../data/asseq/dna_asseq/{args.dataset}/{args.dataset}_dna_ref_500bp.txt"
    dna_alt_asseq = f"../../data/asseq/dna_asseq/{args.dataset}/{args.dataset}_dna_alt_500bp.txt"
    save_dna_repr = f"../../data/feature/dna_repr/{args.dataset}/{args.dataset}_dna_500bp_repr.npy"
    obtain_dna_repr(dna_ref_asseq, dna_alt_asseq, save_dna_repr, args)