import os
import pandas as pd
import argparse

def convert_asseq_line(in_file):
    print("covert asseq to one line")
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
    parser.add_argument("--dataset", default='example')
    parser.add_argument("--cut_len", default=100)
    args = parser.parse_args()

    prt_asseq_path = "./info/prt_asseq/{}_asseq".format(args.dataset)
    save_ref_asseq_path = "./out/{}_prt_ref{}.txt".format(args.dataset,str(args.cut_len*2)) 
    save_alt_asseq_path = "./out/{}_prt_alt{}.txt".format(args.dataset,str(args.cut_len*2)) 
    obtain_cut_prt(prt_asseq_path,save_ref_asseq_path,save_alt_asseq_path)
