import os
import pandas as pd
import argparse

def convert_asseq_line(in_file):
    tmp_file = open(in_file+"_line", 'a')
    seq_line = ""
    flg = 1   

    with open(in_file, 'r') as f:
        for line in f.readlines():
            if line[0] != '>':
                seq_line += line.split('\n')[0]
            else:
                if flg == 1:
                    tmp_file.writelines(str(line))
                    flg = 0
                else:
                    tmp_file.writelines(str(seq_line) + '\n')
                    seq_line = ""
                    tmp_file.writelines(str(line))
        tmp_file.writelines(str(seq_line))
    tmp_file.close()


def obtain_cut_prt(prt_asseq, save_ref_asseq,save_alt_asseq):    
    convert_asseq_line(prt_asseq)

    prt_aaseq = pd.read_csv(prt_asseq + "_line",header=None,engine='python')     
    prt_ref_list, prt_alt_list = [], []

    i=0
    print(len(prt_aaseq))
    while(i < len(prt_aaseq)): 
        ref_line_code = prt_aaseq.iloc[i, 0].split(" ")[0]
        alt_line_code = prt_aaseq.iloc[i+2, 0].split(" ")[0]

        if ref_line_code == alt_line_code:
            prt_altering = str(prt_aaseq.iloc[i+2, 0]).split(" ")
            prt_ref = str(prt_aaseq.iloc[i+1, 0])[:-1]
            prt_alt = str(prt_aaseq.iloc[i+3, 0])[:-1]    

            if prt_altering[4] == 'protein-altering':
                pos_altering = int(prt_altering[7].split("-")[0]) - 1
                start = max(0,pos_altering - args.cut_len)
                end = min(len(prt_ref),pos_altering + args.cut_len)
                prt_ref_cut = prt_ref[start:end]
                prt_alt_cut = prt_alt[start:end]     
                
            elif prt_altering[4] == 'immediate-stoploss':
                pos_altering = int(prt_altering[7].split("-")[0]) - 1
                start = max(0,pos_altering - args.cut_len)
                end1 = min(len(prt_ref),pos_altering + args.cut_len)
                end2 = min(len(prt_alt),pos_altering + args.cut_len)
                prt_ref_cut = prt_ref[start:end1]
                prt_alt_cut = prt_alt[start:end2] 
            
            elif prt_altering[4] == 'immediate-stopgain':
                pos_stop_start = int(prt_altering[7].split("-")[0]) - 1
                start = max(0, pos_stop_start - args.cut_len)
                end = min(len(prt_ref), pos_stop_start + args.cut_len)
                prt_ref_cut = prt_ref[start:end]
                prt_alt_cut = prt_alt[start:pos_stop_start]

            elif prt_altering[4] == 'startloss':
                index = prt_ref.find(prt_alt)
                start = max(0,index - args.cut_len)
                end = min(len(prt_ref),index + args.cut_len)
                prt_ref_cut = prt_ref[start:end]
                prt_alt_cut = prt_alt[0:args.cut_len]  
            
            elif prt_altering[4] == 'silent':
                pos_altering = int(prt_altering[3].split(".")[1][1:-1]) - 1
                start = max(0,pos_altering - args.cut_len)
                end = min(len(prt_ref),pos_altering + args.cut_len)
                prt_ref_cut = prt_ref[start:end]
                prt_alt_cut = prt_alt[start:end]
            
            prt_ref_list.append(prt_ref_cut)
            prt_alt_list.append(prt_alt_cut)
            i = i + 4
        else:
            i = i + 2

    assert len(prt_ref_list) == len(prt_alt_list)

    with open(save_ref_asseq, "w") as f:
        f.writelines(f"{prt_ref_list[i]}\n" for i in range(len(prt_ref_list)))

    with open(save_alt_asseq, "w") as f:
        f.writelines(f"{prt_alt_list[i]}\n" for i in range(len(prt_alt_list)))
    
    os.remove(prt_asseq + "_line")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument("--dataset", default='example', choices=['example','OncoKB','CIViC','CGI','JAX'])
    parser.add_argument("--cut_len", default=100)
    args = parser.parse_args()

    prt_asseq_path = f"../../data/asseq/prt_asseq/{args.dataset}/{args.dataset}_asseq"
    save_ref_asseq_path = f"../../data/asseq/prt_asseq/{args.dataset}/{args.dataset}_asseq_prt_ref_200aa.txt"
    save_alt_asseq_path = f"../../data/asseq/prt_asseq/{args.dataset}/{args.dataset}_asseq_prt_alt_200aa.txt"
    obtain_cut_prt(prt_asseq_path, save_ref_asseq_path, save_alt_asseq_path)
