import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def standardize_transcriptome_data(dataframes):
    """
    对一组转录组测序数据的DataFrame进行标准化处理。如果数据是整数，则先进行log(X+1)转换。

    参数:
    - dataframes: 包含多个DataFrame的列表

    返回:
    - standardized_dataframes: 标准化后的DataFrame列表
    """
    standardized_dataframes = []
    scaler = StandardScaler()
    for df in dataframes:
        df_log = np.log1p(df)
        standardized_df = pd.DataFrame(scaler.fit_transform(df_log), columns=df.columns, index=df.index)
        standardized_dataframes.append(standardized_df)
    print('INFO:: Standardize transcriptome data Successfully.')
    return standardized_dataframes

def load_h5_datasets(file_path):
    """
    This function checks if a given file is an HDF5 file,
    and if so, it loads the specified X into a list.

    Parameters:
    - file_path: str, path to the HDF5 file
    - dataset_names: list of str, names of X to load from the file

    Returns:
    - List of DataFrames containing the data from the specified X
    """

    # Check if the file is an HDF5 file
    if not os.path.isfile(file_path) or not file_path.endswith('.h5'):
        raise ValueError(f"The file '{file_path}' is not a valid HDF5 file.")

    # Initialize a list to store the loaded DataFrames
    dataframes = []
    with pd.HDFStore(file_path, 'r') as store:
        dataset_names = store.keys()
    # Open the HDF5 file and load the specified X
    with pd.HDFStore(file_path) as store:
        for name in dataset_names:
            if name in store.keys():
                dataframes.append(store[name])
            else:
                raise ValueError(f"The dataset '{name}' does not exist in the file '{file_path}'.")
    print("INFO:: Datasets loaded Successfully.")
    return dataframes, dataset_names

def align_and_trim_dataframes(dataframes):
    """
    Align and trim multiple DataFrames by their row indices, keeping only the common rows.

    Parameters:
    - dataframes: list of pd.DataFrame, list of DataFrames to align and trim.

    Returns:
    - A list of DataFrames trimmed to only include the common row indices.
    """
    if not dataframes:
        raise ValueError("The list of DataFrames is empty.")

    # Find the intersection of all row indices
    common_indices = set(dataframes[0].index)
    for df in dataframes[1:]:
        common_indices = common_indices.intersection(df.index)

    if not common_indices:
        raise ValueError("No common row indices found among the DataFrames.")

    # Align and trim each DataFrame to the common indices
    common_indices = sorted(common_indices)
    trimmed_dataframes = [df.loc[common_indices] for df in dataframes]
    print("INFO:: Align and trim dataframes Successfully.")
    return trimmed_dataframes

def check_and_transform_dataframes(dataframes):
    """
    Align multiple DataFrames by their row indices, keep only the common rows,
    convert the DataFrames to ndarrays, transpose them, and return the list of
    transposed arrays along with the row names.

    Parameters:
    - dataframes: list of pd.DataFrame, list of DataFrames to align and convert.

    Returns:
    - A tuple containing:
        - A list of transposed ndarrays.
        - A list of common row names.
    """
    if not dataframes:
        raise ValueError("The list of DataFrames is empty.")

        # Check if all DataFrames have the same number of rows
    num_rows = dataframes[0].shape[0]
    for df in dataframes[1:]:
        if df.shape[0] != num_rows:
            raise ValueError("All DataFrames must have the same number of rows.")

    # Use the row names from any one of the DataFrames (they are all the same)
    row_names = dataframes[0].index

    # Convert each DataFrame to a NumPy array and transpose it
    dataframes = [df.values for df in dataframes]

    print("INFO:: Check and transform dataframes Successfully.")
    return dataframes, row_names

def construt_graph(filename_base_GRN, gene_names):
    if filename_base_GRN is not None:
        TF_TG = pd.read_parquet(filename_base_GRN)
     #   TF_TG = TF_TG[TF_TG['type'] == 'promote']
        adj_matrix = TF_TG.pivot_table(index='TF', columns='Gene', values='weight', fill_value=0)
        adj_matrix = adj_matrix.reindex(index=gene_names, columns=gene_names, fill_value=0)
        print(f"INFO:: Graph({TF_TG.shape[0]}) constructed Successfully.")
        return adj_matrix
    else:
        return None
