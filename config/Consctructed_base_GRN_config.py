import json
import os

class base_GRN_Config:
    def __init__(self):
        self.ref_genome = "hg38"
        self.species = 'human'
        self.out_path = "base_GRN_dataframe.parquet"
        self.prom_thre_coa = 0.7
        self.all_thre_coa = 0.95
        self.scan_tf_batch_size = 30000
        self.scan_tf_n_cpus = 1