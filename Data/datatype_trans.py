import scipy.io
import pandas as pd
import numpy as np
# from scipy.sparse import csr_matrix
import h5py
import os

def read_cicero_h5_GAX(data_folder):
    file = h5py.File(f"{data_folder}/ATAC_GAX.h5", 'r')
    print(list(file.keys()))
    ATAC_data = np.transpose(file['ATAC_GAX'][:])
    ATAC_data_colnames = [name.decode('utf-8') for name in file['ATAC_GAX.colnames'][:]]
    ATAC_data_rownames = [name.decode('utf-8') for name in file['ATAC_GAX.rownames'][:]]
    file.close()
    ATAC_data = pd.DataFrame(ATAC_data, columns=ATAC_data_colnames, index=ATAC_data_rownames)
    return ATAC_data

def save_dcit_to_h5ad(data_dict, save_path, file_label="standard_input.h5"):
    with h5py.File(save_path+file_label, 'w') as hdf_file:
            for df_name, df in data_dict.items():
                if df.dtype.type is np.float32 or df.dtype.type is np.float64:
                    hdf_file.create_dataset(df_name, data=df)
                else:
                    hdf_file.create_dataset(df_name, data=np.array(df.astype('S')))
    print(f'INFO:: h5ad file saved at {save_path+file_label}!')


# 1. Read in matrix/barcode/feature data
data_folder = "./"

indata = scipy.io.mmread(f"{data_folder}/matrix.mtx.gz").tocsc()
cellinfo = pd.read_csv(f"{data_folder}/barcodes.tsv.gz", header=None, sep="\t")
features = pd.read_csv(f"{data_folder}/features.tsv.gz", header=None, sep="\t")
features.columns = ["ENSG symbol", "symbol", "types", "chr", "bp1", "bp2"]

# 2. QC

non_zero_ratio = np.array((indata != 0).sum(axis=0) / indata.shape[0]).flatten()
indata = indata[:, non_zero_ratio >= 0.05]
print(indata.shape)

# 3. Format cell info
cellinfo = cellinfo[non_zero_ratio >= 0.05]
cellinfo.index = cellinfo[0]
cellinfo.columns = ["cells"]
indata = indata.tocsr()
indata.columns = cellinfo.index

# 4. Read features
GEXinfo = features[features['types'] != "Peaks"]
GEXinfo.columns = ["ENSG symbol", "symbol", "types", "chr", "bp1", "bp2"]

# 5. select GEX and GAX
RNA_data = indata[features['types'] != "Peaks", :]
RNA_data = pd.DataFrame(RNA_data.toarray(), columns=indata.columns, index=GEXinfo["symbol"])
RNA_data = RNA_data[~RNA_data.index.duplicated(keep='last')]
ATAC_data = read_cicero_h5_GAX(data_folder)

# 6. All data align
ATAC_data, RNA_data = ATAC_data.align(RNA_data, join='inner')
print(RNA_data.shape)
print(ATAC_data.shape)

# 7. log trans
ATAC_data = np.log1p(ATAC_data)
RNA_data = np.log1p(RNA_data)

# 8. read cluster
RNA_clust = pd.read_csv('GEX_Graph-Based.csv')
cluster_keys = np.unique(RNA_clust['GEX Graph-based'] )
for cluster_key in cluster_keys:
    sub_cluster = RNA_clust['Barcode'][RNA_clust['GEX Graph-based'] == cluster_key]
    subATAC_data = ATAC_data.iloc[:, ATAC_data.columns.isin(sub_cluster.tolist())]
    subRNA_data = RNA_data.iloc[:, RNA_data.columns.isin(sub_cluster.tolist())]
    result = {'ATAC': np.array(subATAC_data),
              'mRNA': np.array(subRNA_data),
              'gene_id': np.array(subRNA_data.index)}
    newdata_folder = os.path.join(data_folder, 'cell_type', cluster_key.replace(' ', '_'))
    if not os.path.exists(newdata_folder):
        os.makedirs(newdata_folder)
    save_dcit_to_h5ad(result, newdata_folder, file_label="/standard_input.h5")











