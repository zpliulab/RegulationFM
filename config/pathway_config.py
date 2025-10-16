
class pathway_Config:
    def __init__(self):
        self.database_path = 'diffusion_model/Building_training_dataset'
        self.database_list = ['RegNetwork', 'DoRothEA', 'KEGG', 'TRRUST']
        #Please select one or more form ['RegNetwork', 'DoRothEA', 'KEGG', 'TTRUST', 'STRING']
        self.kegg_file = 'diffusion_model/Building_training_dataset/kegg/KEGG_all_pathway.pkl'
        self.high_pear_percent = 0.3
        self.high_MI_percent = 0.3
        self.pmi_percent = 0.001
        self.pathway_lim = 300
        self.metacell = True
        self.metacell_num = 200 # 默认200
        self.metacell_k = 5

        # 以下是模拟数据集的参数设置
        self.TP_radio = 0.3
        self.TN_radio = 0.05
        self.seed = 2024
        self.simu = True