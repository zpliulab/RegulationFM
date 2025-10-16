import numpy as np
import pandas as pd
from multiomics_integration.VAE_attention import MultimodalVAE
from multiomics_integration.util import load_h5_datasets, align_and_trim_dataframes, check_and_transform_dataframes, standardize_transcriptome_data, construt_graph
from config import MuInter_config
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import torch.optim as optim
import h5py
import torch
from torch.utils.data import Dataset
import os

class MultiModalDataset(Dataset):
    def __init__(self, datasets, device, max_len=300):
        """
        初始化多模态数据集。
        参数:
            datasets (list): 包含多模态数据的列表。每个元素是一个包含多个模态的元组或列表。
            device (torch.device): 设备（CPU或GPU）用于存储张量。
            max_len (int, optional): 每个模态的最大长度。默认为300。
        """
        self.datasets = datasets
        self.max_len = max_len
        self.device = device
        # 计算每个数据项中每个模态的原始长度
        # 假设所有数据项的模态数量相同
        if len(datasets) > 0:
            self.num_modalities = len(datasets[0])
            self.original_lengths = [
                [len(modality) for modality in dataset]
                for dataset in datasets]
        else:
            self.num_modalities = 0
            self.original_lengths = []

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.datasets)
    def pad_modality(self, modality, max_len):
        """
        对给定模态数据进行填充到最大长度 max_len。

        参数:
            modality (torch.Tensor): 输入的模态张量，形状为 [sequence_length, features]。
            max_len (int): 填充的目标长度。

        返回:
            torch.Tensor: 填充后的模态张量。
        """
        current_len = modality.shape[0]
        if current_len < max_len:
            pad_size = max_len - current_len
            padding = torch.zeros((pad_size, modality.shape[1]), device=self.device)
            modality_padded = torch.cat((modality, padding), dim=0)
        else:
            modality_padded = modality[:max_len]  # 如果超过max_len，截断
        return modality_padded

    def __getitem__(self, idx):
        """
        获取数据集中的第 idx 项。

        参数:
            idx (int): 数据项的索引。

        返回:
            list of torch.Tensor: 填充后的各模态张量列表。
        """
        dataset = self.datasets[idx]
        padded_modalities = []
        for modality in dataset:
            padded = self.pad_modality(modality, self.max_len)
            padded_modalities.append(padded)
        return padded_modalities

    def del_mask(self,dataloader):
        for batch, original_lengths in dataloader:
            for modalities, original_length in zip(batch, original_lengths):
                # 删除填充的0行
                modality_1 = modalities[0][:original_length]
                modality_2 = modalities[1][:original_length]
                modality_3 = modalities[2][:original_length]
                result = [modality_1, modality_2, modality_3]

def z_score_normalization(arr, use_log1p=False):
    if use_log1p:
        arr = np.log1p(arr)  # 对数据进行 log(1 + x) 变换
    # mean = np.mean(arr, axis=0)
    # std = np.std(arr, axis=0)
    # normalized_array = (arr - mean) / std
    min_val = np.min(arr, axis=0)  # 沿着0轴（列）计算最小值
    max_val = np.max(arr, axis=0)  # 沿着0轴（列）计算最大值
    delta = max_val - min_val
    delta[delta == 0] = 1  # 避免除以0
    normalized_array = (arr - min_val) / delta
    return normalized_array


def load_and_check_h5file(file_path):
    with pd.HDFStore(file_path, 'r') as store:
        keys = store.keys()
    data_list = []
    for key in keys:
        df = pd.read_hdf(file_path, key=key)
        if df.isnull().values.any():
            raise ValueError(f"ERROE:: There are NaN values in the data set, located at key: {key}")
        data_list.append(df)
    row_counts = [df.shape[0] for df in data_list]
    assert len(set(row_counts)) == 1, 'ERROE:: The number of rows in each dataset must be the same!'
    return data_list

def extract_number(s):
    # 尝试从字符串中提取数字，如果提取不到则返回 None
    try:
        return int(''.join(filter(str.isdigit, s)))
    except ValueError:
        return None

# 定义一个自定义的排序函数，优先按数字排序，如果没有数字则按字符串排序
def sort_key(s):
    num = extract_number(s)
    # 如果能提取出数字，则返回数字，否则返回原字符串
    return (num if num is not None else s)

def trans_ndarray_format(group, key='network'):
    network_raw = group[key][()]
    network_data = pd.DataFrame({
        'TF': [str(row[0], 'utf-8') if isinstance(row[0], bytes) else row[0] for row in network_raw],
        'Gene': [str(row[1], 'utf-8') if isinstance(row[1], bytes) else row[1] for row in network_raw],
        'weight': [row[2] for row in network_raw]})
    new_dict = {key: np.array(network_data)}  # 将 DataFrame 存入字典
    return new_dict

def read_simulation_h5_file_to_list(file_path):
    structured_data_dict = {}
    with (h5py.File(file_path, 'r') as h5_file):
        group_names_all = sorted(h5_file.keys(), key=sort_key)
        for group_name in group_names_all:
            group = h5_file[group_name]

            feature_id_list = [str(item, 'utf-8') if isinstance(item, bytes) else item for item in
                               group['gene_id'][()]]

            mRNA = group['mRNA'][()]
            if len(feature_id_list) != mRNA.shape[0]:
                mRNA = np.transpose(group['mRNA'][()])
            data_dict = {
                'mRNA': np.array(mRNA),
                'gene_id': np.array(feature_id_list),
            }
            if 'network' in group.keys():
                new_dict = trans_ndarray_format(group, key='network')
                data_dict = {**data_dict, **new_dict}
            if 'KEGG' in group.keys():
                new_dict = trans_ndarray_format(group, key='KEGG')
                data_dict = {**data_dict, **new_dict}
            if 'DoRothEA' in group.keys():
                new_dict = trans_ndarray_format(group, key='DoRothEA')
                data_dict = {**data_dict, **new_dict}
            if 'TRRUST' in group.keys():
                new_dict = trans_ndarray_format(group, key='TRRUST')
                data_dict = {**data_dict, **new_dict}

            non_mrna_keys = [key for key in group.keys() if
                             key not in ['mRNA', 'gene_id', 'network', 'DoRothEA', 'KEGG', 'TRRUST']]
            for other_keys in non_mrna_keys:
                omics = group[other_keys][()]
                if other_keys in ['protein', 'unspliced']:
                    omics = np.log1p(omics)
                if len(feature_id_list) != group['mRNA'][()].shape[0]:
                    omics = np.transpose(omics)
                new_dict = {str(other_keys): np.array(omics)}  # 将 DataFrame 存入字典
                data_dict = {**data_dict, **new_dict}

            structured_data_dict[group_name] = data_dict

    return structured_data_dict


def save_dcit_to_h5ad(data_dict, save_path, file_label="joint_represent_input.h5"):
    new_floder = os.path.join(save_path, file_label)
    with h5py.File(new_floder, 'w') as hdf_file:
        for dataset_name, df_dict in data_dict.items():
            group = hdf_file.create_group(dataset_name)
            for df_name, df in df_dict.items():
                if df.dtype.type is np.float32 or df.dtype.type is np.float64:
                    group.create_dataset(df_name, data=df)
                else:
                    group.create_dataset(df_name, data=np.array(df.astype('S')))

    print(f'INFO:: h5ad file saved at {new_floder}!')


def train_model(model, train_loader, epochs=10, lr=0.001, beta=1.0, device='cpu'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)  # 定义优化器
    model = model.to(device)
    progress_bar = tqdm(range(epochs), ncols=100, desc='Training VAE model')
    model.train()
    for epoch in progress_bar:
        train_loss = 0
        for x_list in train_loader:
            optimizer.zero_grad()
            joint_z, recon_joint_list,primary_mu, primary_logvar,attention_weights = model(x_list)
            loss = model.loss_function(recon_joint_list, x_list, [primary_mu], [primary_logvar], joint_z, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()
        progress_bar.set_postfix({'Epoch': f'{epoch + 1:d}', 'Loss': f'{train_loss / len(train_loader.dataset):.4f}'})


def run_VAE_attention(Multiomics_data_list, args):
    # 每个模态的数据按不同维度传入（多个模态的数据放在list中）
    n_cells = [Multiomics_data.shape[1] for Multiomics_data in Multiomics_data_list[0]]  # 每个模态的数据细胞数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for idx1, data_list in enumerate(Multiomics_data_list):
        for idx2, data in enumerate(data_list):
            #if idx2 == 1:
            if np.max(data) < 10:
                data1 = torch.tensor(z_score_normalization(np.array(data), use_log1p=False), dtype=torch.float32, device=device)
            else:
                data1 = torch.tensor(z_score_normalization(np.array(data), use_log1p=True), dtype=torch.float32, device=device)
            Multiomics_data_list[idx1][idx2] = data1

    # 创建多模态 VAE 模型
    model = MultimodalVAE(n_cells, args.hidden_dim, args.latent_dims).to(device)

    # 训练模型
    # multi_modal_dataset = MultiModalDataset(Multiomics_data_list, device)
    dataloader = DataLoader(Multiomics_data_list, batch_size=32, shuffle=True)
    train_model(model, dataloader, epochs=args.epochs, lr=args.lr, beta=args.beta, device=device)

    # 获得特定模态和联合模态的表示
    new_represent = []
    for data_list in Multiomics_data_list:
        specific_reps, joint_rep = model.get_specific_and_joint_representations(data_list)
        specific_reps.insert(0, joint_rep)  # 将联合表示插入到列表的开头
        new_represent.append(specific_reps)
    return new_represent



def run_Multiomic_learning(ALL_data_file_path, args):
    file_path = os.path.join(ALL_data_file_path, args.input_path)
    All_data_dict = read_simulation_h5_file_to_list(file_path)
    All_data_dict_new = {}
    for idx, (dataset_name, Multiomics_data_dict) in enumerate(All_data_dict.items()):
        print(f'INFO:: Calculating VAE-attention embedding of {idx + 1}-th datasets!')
        temp_data_dict = Multiomics_data_dict.copy()
        if 'network' in temp_data_dict.keys():
            del temp_data_dict['network']
        if 'DoRothEA' in temp_data_dict.keys():
            del temp_data_dict['DoRothEA']
        if 'KEGG' in temp_data_dict.keys():
            del temp_data_dict['KEGG']
        if 'TRRUST' in temp_data_dict.keys():
            del temp_data_dict['TRRUST']
        if 'gene_id' in temp_data_dict.keys():
            del temp_data_dict['gene_id']

        keys_of_interest = ['mRNA', 'scRNA', 'RNA']
        # 使用 next() 找到第一个符合条件的 value，如果没有找到则返回 None
        target_value = next((temp_data_dict[key] for key in keys_of_interest if key in temp_data_dict), None)
        # 提取所有的值
        all_values = list(temp_data_dict.values())
        # 如果找到目标 value，则移到最前面
        if target_value is not None:
            all_values = [target_value] + [value for value in all_values if not (
                    isinstance(value, np.ndarray) and isinstance(target_value, np.ndarray) and
                    value.shape == target_value.shape and (value == target_value).all())]
        else:
            print("No key in 'mRNA', 'scRNA' and 'RNA'")
        output = run_VAE_attention([all_values], args)

        # 将 Multiomics_data_dict 学习联合表示
        All_data_dict[dataset_name]['share'] = output[0][0]
        for idx, omicnames in enumerate(list(temp_data_dict.keys())):
            All_data_dict[dataset_name][omicnames + '_new'] = output[0][idx]
        All_data_dict_new[dataset_name] = All_data_dict[dataset_name]
    save_dcit_to_h5ad(All_data_dict_new, ALL_data_file_path, file_label=args.out_path)
    return All_data_dict


def Multiomics_learning(ALL_data_file_path, rerun=False, flabel_in=None, flabel_out=None):
    '''
    这是一个构建多组学数据联合表示的函数
    :param ALL_data_file_path: 多组学数据的路径，确保存在cicero_output/base_GRN_dataframe.parquet 和 standard_input.h5 文件
    :param rerun:
    :return:
    '''
    args = MuInter_config()
    if flabel_in is not None:
        args.input_path = flabel_in
    if flabel_out is not None:
        args.out_path = flabel_out
    if rerun:
        _ = run_Multiomic_learning(ALL_data_file_path, args)
    # All_data = load_and_check_h5file(ALL_data_file_path + args.out_path)
    return os.path.join(ALL_data_file_path, args.out_path)


if __name__ == "__main__":
    ALL_data_file_path = '/home/wcy/RegulationGPT/Data/Simulation_L'
    Multiomics_learning(ALL_data_file_path, rerun=True)

    import subprocess
    def run_script_in_different_conda_env(script_path, env_name):
        """
        在指定的conda环境中运行Python脚本。
        :param script_path: 要运行的脚本的路径
        :param env_name: 目标conda环境的名称
        """
        # 构建在指定conda环境中执行Python脚本的命令
        command = f"conda run -n {env_name} python {script_path}"
        # 使用subprocess.run执行命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # 打印输出结果和错误信息
        print("输出:")
        print(result.stdout)
        print("错误:")
        print(result.stderr)
    # 示例使用
    run_script_in_different_conda_env('/home/wcy/RegulationGPT/Data/Simulation_100/main_GRNBoost2_simulation_dyngen.py', 'SCENIC')


