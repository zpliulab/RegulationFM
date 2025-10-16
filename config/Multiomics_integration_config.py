import json
import os

class MuInter_config:
    def __init__(self):
        # Path and file related configuration
        self.latent_dims = 1024  # latent representation dimension of VAE
        self.hidden_dim = 512
        self.epochs = 500  # number of epochs to train VAE
        self.batch_size = 1024
        self.lr = 1e-4  # learning rate
        self.beta = 1000  # parameter of VAE_KL
      #  self.base_GRN_path = "/cicero_output/base_GRN_dataframe.parquet" # base_GRN path
        self.input_path = "standard_input.h5"  # input path
        self.out_path = "joint_represent_input.h5"  # output path
        self.simu = False