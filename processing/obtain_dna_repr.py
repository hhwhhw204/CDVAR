
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging
logging.set_verbosity_error()


def process_features(model, tokenizer, sequence):
    return [model(tokenizer(seq, return_tensors='pt')["input_ids"].cuda())[0].squeeze(0).cpu().detach().numpy() 
            for seq in tqdm(sequence)]


def obtain_dna_repr(ref_asseq_file, alt_asseq_file, save_dna_repr, args):
    # Load model. 
    # The model reference: https://huggingface.co/zhihan1996/DNABERT-2-117M
    tokenizer = AutoTokenizer.from_pretrained("DNABERT_2_117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("DNABERT_2_117M", trust_remote_code=True).cuda()
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
    parser.add_argument("--dataset", default='example')
    parser.add_argument("--seq_len", default=250)
    args = parser.parse_args()

    dna_ref_asseq = "./out/example_dna_ref_{}bp.txt".format(args.seq_len*2)
    dna_alt_asseq = "./out/example_dna_alt_{}bp.txt".format(args.seq_len*2)
    save_dna_repr = "./out/eample_dna_{}bp_repr.npy".format(args.seq_len*2)
    obtain_dna_repr(dna_ref_asseq, dna_alt_asseq, save_dna_repr, args)