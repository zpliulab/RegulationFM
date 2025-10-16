import json
import os

class main_Config:
    def __init__(self):
        # Path and file related configuration
        self.save_dir = "result"                   # Result saving directory
        self.save_label = "Training_data"           # Save label
        self.test_pathway = None
        self.num_train_datasets = 200
        self.num_val_datasets = [200, 204]          # 请输入验证集的编号，或者symbol，例如'hsa05224'
        self.val_datasets_path = None               # 如果不设定验证集路径，默认为训练集同一文件

        # Model related configuration
        self.diffusion_timesteps = 1000             # Diffusion time steps
        self.ensemble = 30                          # Number of ensemble learning
        self.max_nodes = None                       # Maximum number of nodes
        self.show = False                           # Whether to display the results

        # Training related configuration
        self.num_epoch = 600                        # Number of training epoch
        self.batch_size = 24                        #   Batch size
        self.LR = 1e-3                               # Learning rateLearning rate
        self.test_interval = 200                    # Test interval
        self.save_interval = 200                    # (checkponit)Save interval
        self.n_rep = 3                              # Number of repetitions
        self.n_job = 1               # Number of parallel jobs

        # Network structure related configuration
        self.num_layer = 2                          # Number of network layers
        self.num_head = 4                           # Number of attention heads
        self.num_MLP = 32                           # Number of MLP layer nodes
        self.num_GTM = 16                           # Number of graph transformer layer nodes
        self.cell_type_pathway_dict = {'Naive_CD4+T': 'hsa04660',
                                       'CD14+Mono': 'hsa04620',
                                       'Memory_CD4+T': 'hsa04660',
                                       'NKcell': 'hsa04650',
                                       'CD8+T': 'hsa04660',
                                       'Bcell': 'hsa04662',
                                       'FCGR3A+Mono': 'hsa04666',
                                       'DC': 'hsa04612',
                                       'IL7R+Memory_CD4+T': 'hsa04660',
                                       'Platelet': 'hsa04611',
                                       'BG': 'hsa04724',
                                       'GC': 'hsa04720',
                                       'OL': 'hsa04350',
                                       'Microglia': 'hsa04062',
                                       'DSLL_state1': 'hsa05220',
                                       'DSLL_state2': 'hsa05220',
                                       'Naive_CD8+T': 'hsa04660',
                                       'Exhausted_CD8+T': 'hsa04750',
                                       'Exhausted_CD4+T': 'hsa04750',
                                       'Effector_like_CD8+T': 'hsa04640',
                                       'Macrophage': 'hsa04620'
                                       }
        # Database and LLM related configuration
        self.net_key_par = {'Flag_base_GRN': True}   # 是否打开

# # Create config folder
# config_folder = 'config'
# os.makedirs(config_folder, exist_ok=True)
#
# # Define config file path
# config_file_path = os.path.join(config_folder, 'config.json')
#
# # write config file
# with open(config_file_path, 'w') as f:
#     json.dump(config, f, indent=4)
#
# print(f" Configuration file  '{config_file_path}' has been created.")
