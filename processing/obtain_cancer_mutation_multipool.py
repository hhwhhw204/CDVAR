import pandas as pd
import pybedtools
from tqdm import tqdm
import concurrent.futures
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_pancan_fea_by_row(row,flank,pancan,columns):
    chrom = row["chr"]
    pos = row["pos"]
    end = pos + 1
    pos -= flank
    if pos < 0: pos = 0
    end += flank

    string = str(chrom) + "\t" + str(pos) + "\t" + str(end) + "\n"
    mut = pybedtools.BedTool(string, from_string=True)
    relat_pancan = pancan.intersect(mut, u=True)
    relat_df = pd.read_table(relat_pancan.fn, names=columns)
    new_row = pd.Series(dtype='object')

    for col in columns[3:]:
        count = relat_df[col].value_counts()
        new_row = pd.concat([new_row, count])
    new_row = pd.Series([chrom,row["pos"]], index=["chr","pos"])._append(new_row)
    return new_row



def get_pancan_anno_multi_thre(file_name):
    columns = ["Chromosome", "Start_Position", "End_Position", "Variant_Classification", "Variant_Type",
            "Consequence", "BIOTYPE", "IMPACT", "VARIANT_CLASS"]
    pancan = pybedtools.BedTool('/PanCanAtlas.bed')
    pancan_df = pd.read_table(pancan.fn, names=columns)

    # Get column names for all features  
    total_list = []
    total_list = [x for col in columns[3:] for x in pancan_df[col].unique()]

    # Read my mutation data
    data_df = pd.read_csv( "info/csv/" + file_name + ".csv", sep=",", header=None)
    data_df.columns = ["chr", "pos", "ref", "alt"]

    # bed intersect
    flank = 100
    block = 500
    k = int(data_df.shape[0]/block)
    for z in tqdm(range(0,k+1)):
        start = z * block
        end = (z+1) * block
        print(str(start)+"---------"+str(end))
        if end > data_df.shape[0]:
            end = data_df.shape[0]

        futures = {}
        fea_df_dict = {}
        fea_df = pd.DataFrame(columns=total_list)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for idx, row in data_df.iloc[start:end].iterrows():
                future = executor.submit(get_pancan_fea_by_row, row,flank,pancan,columns)
                futures[future] = idx

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                new_row = future.result()
                fea_df_dict[idx] = new_row

        for idx in range(len(fea_df_dict)):
            new_row = fea_df_dict[idx+z*block]
            fea_df = fea_df._append(new_row,ignore_index=True)
            fea_df.fillna(0,inplace=True)

        fea_df.to_csv(f"./out/{file_name}_{flank}bpx2_fea_{z}x{block}.tsv", sep="\t", header=True, index=False)
        print("intersect complete")



if __name__=='__main__':
    file_name = "example"
    get_pancan_anno_multi_thre(file_name)