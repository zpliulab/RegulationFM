from scATAC_co_accessibility import from_ATAC_constructed_base_GRN
from multiomics_integration import Multiomics_learning
from config.RegulationGPT_config import main_Config as config
from diffusion_model import RegulationGPT
import torch
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from diffusion_model.discrete.diffusion_utils import find_denominator_element

if __name__ == '__main__':
    # ------------------------------
    # Main Configuration
    # ------------------------------
    '''
        Before running RegulationGPT, you must standardize your data into Gene*Cell format.
        To ensure proper encoding, store multiple omics datasets in a single HDF5 file 
        (named "standard_input.h5"), which should contain at least one dataset named "mRNA"
        and one named "Gene_id".

        ATAC data should be processed using Signac or Cicero to generate the gene activity matrix.
    '''
    ALL_data_file_path = '/home/wcy/RegulationGPT/Data/Simulation_L'

    # Load configuration parameters
    args = config()
    args.num_epoch = 200 # Number of training epochs

    # ------------------------------
    # Step 1: Build joint representation from multi-omics data
    # ------------------------------
    # Before running this step, you must convert raw data into standard_input.h5 using datatype_trans.py
    train_filename = Multiomics_learning(ALL_data_file_path, rerun=False)

    # ------------------------------
    # Step 2: Split large dataset into smaller graphs for training/testing
    # ------------------------------
    # You can set options here to extract both training and test datasets.
    # This process is mainly controlled through the RegulationGPT class.
    args.test_interval = 100  # Frequency of testing during training
    args.save_interval = 100  # Frequency of checkpoint saving
    args.n_job = 1            # Number of parallel jobs
    trainer = RegulationGPT(args)  # Initialize RegulationGPT trainer
    training = False     # If 'training' is set to True, the model will be trained from scratch
    if training:
        best_mean_AUC, train_model_file, printf = trainer.train(train_filename, base_GRN_link=None)

    # ------------------------------
    # Step 3: Load Pretrained Model and Prepare Test Data
    # ------------------------------
    train_model_file = 'results_checkpoint/Training_data/checkpoint_20250604-145130_epoch100.pth'
    ALL_data_file_path = '/home/wcy/RegulationGPT/Data/Simulation_test/'      # Load simulation test dataset (gene expression profiling)
    test_filename = Multiomics_learning(ALL_data_file_path, rerun=False)

    # If you want to build a network for a specific gene set or KEGG pathway,
    # you can specify KEGG IDs or file paths here. Otherwise, set to None.
    args.test_pathway = None

    # Load pretrained diffusion model checkpoint
    diffusion_pre = torch.load(train_model_file, map_location=trainer.device)

    # ------------------------------
    # Step 4: Model Evaluation Settings
    # ------------------------------
    result_filename = 'Simulation_0604.data'
    results = {'AUC': [], 'AUPR': [], 'Ep': [], 'Epr': [], 'F1': [], 'nodenum': []}
    par = tqdm(range(100), ncols=100)
    for test_num in par:
        # Load test data and corresponding ground-truth network labels
        testdata, truelabel = trainer.load_test_data(test_filename, num=test_num, network_key_list='network')
        # Perform inference using the pretrained RegulationGPT model
        adj_final = trainer.test(diffusion_pre, testdata, truelabel)  # 训练模拟数据

        # Thresholding: remove weak edges to sparsify the GRN
        if adj_final is not None:
            adj_final[adj_final < find_denominator_element(adj_final, threshold=0.2)] = 0  # threshold 越小 边越少
            performance = trainer.evaluation(adj_final, truelabel)
            results['AUC'].append(performance['AUC'])
            results['AUPR'].append(performance['AUPR'])
            results['F1'].append(performance['F1'])
            results['Ep'].append(performance['Ep'])
            results['Epr'].append(performance['Epr'])
            results['nodenum'].append(testdata.x.shape[0])
            f = open(result_filename, 'wb')
            pickle.dump(results, f)
            f.close()
            par.set_description(
                f"The: {test_num + 1:d}-th  GEX. RegulationGPT*-AUROC: {performance['AUC']:.4f} "
                f"AUPRC: {performance['AUPR']:.4f} "
                f"Ep: {performance['Ep']:.4f} "
                f"Epr: {performance['Epr']:.4f} "
                f"F1-score: {performance['F1']:.4f}")
        else:
            results['AUC'].append(0)
            results['AUPR'].append(0)
            results['F1'].append(0)
            results['Ep'].append(0)
            results['Epr'].append(0)
            results['nodenum'].append(0)
    # ------------------------------
    # Compute and display global averages
    # ------------------------------
    golbalmeanAUC = np.mean(results['AUC'])
    golbalmeanAUPR = np.mean(results['AUPR'])
    golbalmeanF1 = np.mean(results['Epr'])

    # Print overall model evaluation summary
    print(
        f"DiffGRN*-AUC : {golbalmeanAUC:.4f} AUPR mean: {golbalmeanAUPR:.4f}  EPR mean: {golbalmeanF1:.4f}")