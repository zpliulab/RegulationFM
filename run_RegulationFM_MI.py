import copy
import pandas as pd
import numpy as np
from scATAC_co_accessibility import from_ATAC_constructed_base_GRN
from multiomics_integration import Multiomics_learning
from config.RegulationGPT_config import main_Config as config
from config import pathway_Config

from diffusion_model import RegulationGPT
from diffusion_model.Building_training_dataset.pathway_segmentation import Segmentation_pathway
import torch
from Data.Standard_10X_RNA_ATAC import Standard_10X_cicero_file
import os
from diffusion_model.discrete.diffusion_utils import find_denominator_element
from Data.create_joint_cache import create_all_type_training_data
import pickle
import gzip


def get_subfolders(path):
    """
    Traverse the given directory path and return a list of all subfolder paths.
    Args:
        path (str): Root directory to search for subfolders.
    Returns:
        list: A list containing full paths of all subdirectories found.
    """
    subfolders = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            subfolders.append(os.path.join(root, dir_name))
    return subfolders


if __name__ == '__main__':
    # ------------------------------
    # Main Configuration
    # ------------------------------
    '''
        Before running RegulationGPT, you must standardize your data into a Gene*Cell format.
        To ensure proper encoding, store multiple omics datasets in a single HDF5 file 
        (named "standard_input.h5"), which should contain at least one dataset named "mRNA" 
        and one named "gene_id".

        ATAC data should be processed using Signac or Cicero to generate the gene activity matrix.
    '''
    All_data_folder = '/data/wcy_data/RegulationGPT_key/myocardial_fibro/GSM8352052_MA9'
    # ------------------------------
    # Step 1: Load main configuration and pathway settings
    # ------------------------------
    args = config()
    pathway_args = pathway_Config()
    pathway_args.metacell_num = 50 # Define the number of metacells for pathway segmentation

    # ------------------------------
    # Step 2: Flags for rerunning intermediate steps
    # ------------------------------
    need_rerun_baseGRN = False
    need_Segmentation_pathway = False
    need_Multiomics_learning = False
    need_rerun_cicero = False
    need_evaluation_database = ['network', 'DoRothEA', 'KEGG', 'TRRUST'] # List of network databases used for evaluation
    unique_cell_type = ["Myeloid", "Cardiomyocyte", "Pericyte", "Adipocyte",'Fibroblast',"Endothelium","Endocardium"] # Define the unique cell types to be analyzed

    # ------------------------------
    # Step 3: Set parameters and train model for all cell types
    # ------------------------------
    args.save_label = 'Myocardial_All_sample'
    args.n_job = 1
    args.num_epoch = 50
    args.max_nodes = 500
    trainer = RegulationGPT(args) # Initialize RegulationGPT trainer
    # Path to a pretrained RegulationFM model
    train_model_file = 'results_checkpoint/Myocardial_infarction/Myocardial_infarction_epoch50_train195.pth'

    # Iterate over each cell type for testing
    for select_cell_type in unique_cell_type:
        args.test_pathway = '/data/wcy_data/RegulationGPT_key/myocardial_fibro/DEG_top500_by_cluster'

        args.test_pathway = os.path.join(args.test_pathway, 'DEG_' + select_cell_type + '.csv')
        cluster_file_path = os.path.join(All_data_folder, 'cicero_output', select_cell_type)

        # Determine whether to rerun base GRN or Cicero preprocessing
        need_rerun_baseGRN= (not os.path.exists(
            os.path.join(cluster_file_path, "all_peaks.csv")) or need_rerun_baseGRN)
        need_rerun_cicero = (not os.path.exists(
            os.path.join(cluster_file_path, "ATAC_GAX_RNA_GEX.h5")) or need_rerun_cicero)
        base_GRN_link = from_ATAC_constructed_base_GRN(All_data_folder, select_cell_type, rerun=need_rerun_baseGRN,
                                                       run_R_code=need_rerun_cicero)     # Construct the baseline GRN from scATAC data
        if not os.path.exists(os.path.join(cluster_file_path, "standard_input.h5")):
            Standard_10X_cicero_file(All_data_folder, unique_cell_type=select_cell_type)
        Segmentation_pathway(cluster_file_path, test_pathway=args.test_pathway,
                             test=True,
                             filelabel='standard_input_pathway_test.h5',
                             multi_test_label=True,
                             args=pathway_args)
        need_Multiomics_learning_TE = (not os.path.exists(
            os.path.join(cluster_file_path, "joint_represent_pathway_test.h5")) or need_Multiomics_learning)
        test_filename = Multiomics_learning(cluster_file_path,
                                            rerun=need_Multiomics_learning_TE,
                                            flabel_in="standard_input_pathway_test.h5",
                                            flabel_out="joint_represent_pathway_test.h5")
        args.val_datasets_path = test_filename

        # ------------------------------
        # Step 4: Model Testing
        # ------------------------------
        diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
        # Load test data and true labels
        testdata, truelabel = trainer.load_test_data(test_filename,
                                             num='default',
                                             base_GRN_link=base_GRN_link)  # base_GRN_link is required!
        # Perform inference using the pretrained model
        adj_final = trainer.test(diffusion_pre, testdata)
        # Evaluate model performance against multiple benchmark networks
        for network_key in need_evaluation_database:
            performance = trainer.evaluation(adj_final, truelabel[network_key], flag=True)
            p1 = (f" RegulationGPT*-AUROC: {performance['AUC']:.4f} "
                  f"AUPRC: {performance['AUPR']:.4f} "
                  f"Ep: {performance['Ep']:.4f} "
                  f"Epr: {performance['Epr']:.4f} "
                  f"F1-score: {performance['F1']:.4f} "
                  f"CC: {sum(performance['CC'].values())/len(performance['CC']):.4f} "
                  f"EC: {sum(performance['EC'].values())/len(performance['EC']):.4f} " )
            print(p1)

        # Save the predicted GRN as a pickle file for downstream analysis
        test_GRN_file = os.path.join(cluster_file_path, 'GRN_by_MIFM_MA93.pkl')
        with open(test_GRN_file, "wb") as file:
            pickle.dump(adj_final, file)