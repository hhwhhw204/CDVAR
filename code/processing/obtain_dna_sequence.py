import pandas as pd
import pysam
import argparse
import re
from tqdm import tqdm

def save_to_file(data, file):
    with open(file, "w") as f:
        for line in data:
            f.write(line + "\n")


def obtain_dna_cut(csv, hg19_file, dna_ref_save, dna_alt_save):   
    df = pd.read_csv(csv, sep=",", header=None)
    df.columns = ["chr", "start", "ref", "alt"]
    df['chr'] = df['chr'].apply(lambda x: "chr"+str(x).replace("chr", ""))

    fastafile=pysam.FastaFile(hg19_file)    
    ref_seqs, alt_seqs, ref_lines, alt_lines = [], [], [], []

    for _, row in tqdm(df.iterrows()):
        seq = fastafile.fetch(row['chr'], row['start']-args.seq_len, row['start']+args.seq_len).upper()
        assert seq[args.seq_len-1] == row['ref'], "error!"
        ref_seq = seq[:args.seq_len-1] + row['ref'] + seq[args.seq_len:]
        alt_seq = seq[:args.seq_len-1] + row['alt'] + seq[args.seq_len:]
        ref_seqs.append(re.sub(r'[^TCGA]', "A", ref_seq))
        alt_seqs.append(re.sub(r'[^TCGA]', "A", alt_seq))
        ref_lines.append(f"{row['chr']}_{row['start']}_{row['ref']}_{row['alt']}\t{ref_seq}")
        alt_lines.append(f"{row['chr']}_{row['start']}_{row['ref']}_{row['alt']}\t{alt_seq}")
    save_to_file(ref_lines, dna_ref_save)
    save_to_file(alt_lines, dna_alt_save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument("--dataset", default='CGI', choices=['example','OncoKB','CIViC','CGI','JAX'])
    parser.add_argument("--seq_len", default=250)
    args = parser.parse_args()

    csv = f"../../data/input/{args.dataset}/{args.dataset}.csv"
    hg19_file = "../../data/utils/hg19.fa"
    save_dna_ref_asseq = f"../../data/asseq/dna_asseq/{args.dataset}/{args.dataset}_dna_ref_500bp.txt"
    save_dna_alt_asseq = f"../../data/asseq/dna_asseq/{args.dataset}/{args.dataset}_dna_alt_500bp.txt"
    obtain_dna_cut(csv,hg19_file,save_dna_ref_asseq,save_dna_alt_asseq) 
