import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from diffusion_model.discrete.models import Bagging_ridge_net
from config import pathway_Config
from diffusion_model.Building_training_dataset.simulation import modify_matrix, calculate_auc
import h5py
import numbers
def crop_network(node_feature, sub_base_GRN):
    genenum = node_feature.shape[0]
    genes = [f'G{i}' for i in range(genenum)]
    node_feature = pd.DataFrame(np.transpose(node_feature), columns=genes)
    Br = Bagging_ridge_net()
    linkList = Br.run_bagging_ridge(node_feature, genes, np.array(sub_base_GRN))
    adj_matrix = pd.DataFrame(np.zeros((genenum, genenum)),
                              index=genes, columns=genes)
    for idx, row in linkList.iterrows():
        adj_matrix.loc[row['source'], row['target']] = 1
    adj_matrix = adj_matrix.to_numpy()
    adj_matrix = np.transpose(adj_matrix)
    return adj_matrix


# 辅助函数： 把邻接矩阵转换为张量
def adj2tensor(adj_pmi_ppc_mean, sc_pc_adj_matrix=None):
    if sc_pc_adj_matrix is None:
        sc_pc_adj_matrix = adj_pmi_ppc_mean
    sc_pc_adj_matrix = torch.tensor(sc_pc_adj_matrix, dtype=torch.float32)
    adj_pmi_ppc_mean = torch.tensor(adj_pmi_ppc_mean, dtype=torch.float32)
    crop_indices_tensor = sc_pc_adj_matrix.nonzero(as_tuple=False).t().contiguous()  # 从张量sc_pc_adj_matrix中提取非零值的索引
    crop_indices_values = adj_pmi_ppc_mean[
        crop_indices_tensor[0, :], crop_indices_tensor[1, :]]  # 从张量adj_pmi_ppc_mean中提取非零值
    if len(crop_indices_values) == 0:
        crop_indices_tensor = torch.tensor([[0], [0]])
        crop_indices_values = torch.tensor([0])
    return crop_indices_tensor, crop_indices_values

def Extract_base_GRN(base_GRN_list, gene_list):
    base_GRN_list = base_GRN_list[
        base_GRN_list['TF'].isin(gene_list) & base_GRN_list['Gene'].isin(gene_list)]
    gene_set = set(gene_list)
    connections = {gene: {} for gene in gene_list}  # 将字典的值改为字典，以存储每个TF的连接类型
    adj_matrix = pd.DataFrame(0, index=gene_list, columns=gene_list)

    # 判断是否包含 'type' 列
    if 'type' in base_GRN_list.columns:
        type_mapping = {
            'promote': 1,
            'All': 2,
            'public': 3
        }
        # 迭代 DataFrame，只处理两端都在 gene_list 中的连接
        for _, row in base_GRN_list.iterrows():
            gene, tf, type_value = row['Gene'], row['TF'], row['type']
            if gene in gene_set and tf in gene_set:
                connections[gene][tf] = type_mapping.get(type_value, 0)  # 如果类型不存在于映射中，则设为0
        # 将连接类型写入邻接矩阵
        for gene, tfs in connections.items():
            for tf, type_value in tfs.items():
                adj_matrix.loc[gene, tf] = type_value
    else:
        # 如果不包含 'type' 列，直接将连接标记为 1
        for _, row in base_GRN_list.iterrows():
            gene, tf = row['Gene'], row['TF']
            if gene in gene_set and tf in gene_set:
                adj_matrix.loc[gene, tf] = 1

    adj_matrix = adj_matrix.T
    return adj_matrix

def crop_network(node_feature, sub_base_GRN):
    genenum = node_feature.shape[0]
    genes = [f'G{i}' for i in range(genenum)]
    node_feature = pd.DataFrame(np.transpose(node_feature), columns=genes)
    Br = Bagging_ridge_net()
    linkList = Br.run_bagging_ridge(node_feature, genes, np.array(sub_base_GRN))
    adj_matrix = pd.DataFrame(np.zeros((genenum, genenum)),
                              index=genes, columns=genes)
    for idx, row in linkList.iterrows():
        adj_matrix.loc[row['source'], row['target']] = 1
    adj_matrix = adj_matrix.to_numpy()
    adj_matrix = np.transpose(adj_matrix)
    return adj_matrix



def matrix2Data(adj_matrix, node_feature, Previous_data =None, log_trans=False, base_GRN_list=None, mulit_test_label=True):
    '''
        该函数用于将输入的邻接矩阵和节点特征矩阵构建为Data结构
    :param adj_matrix: 邻接矩阵
    :param node_feature: 节点特征
    :param log_trans: 是否需要log转换
    :param theta: 相关参数字典
    :param Adddatabse: 
    :param base_GRN_list: 
    :return: 
    '''
    # node_feature = np.array(node_feature)
    if adj_matrix is not None:
        adj_matrix = np.array(adj_matrix)
        if node_feature.shape[0] != adj_matrix.shape[0]:
            node_feature = np.transpose(node_feature)
    x = torch.tensor(np.array(node_feature), dtype=torch.float)

    # 对数变换
    if log_trans:
        x = torch.log1p(x)  # 使用 log1p 进行对数变换，避免零值的问题

    # 归一化
    scaler = StandardScaler()
    x_normalized = torch.tensor(scaler.fit_transform(x), dtype=torch.float)

    if Previous_data is not None:
        existing_x_count = sum(1 for key in Previous_data.keys() if key.startswith('x'))
        x_key = f'x_{existing_x_count}'
        Previous_data[x_key] = x_normalized
        return Previous_data

    if adj_matrix is not None:
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        indices_tensor = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        num_edges = indices_tensor.shape[1]
        values_tensor = torch.ones(num_edges, dtype=torch.float32)
        assert indices_tensor.shape[1] == values_tensor.shape[0], "Gold standard：边的数量与权重出现错误！"
        assert indices_tensor.max() < x_normalized.shape[0], "Gold standard：边的最大索引超出表达谱范围！"
    else:
        indices_tensor = None
        values_tensor = None

    # 增加一些其他网络/相关性信息，以便于后续使用
    base_GRN_indices_tensor = None
    base_GRN_indices_values = None

    if base_GRN_list is not None:
        if base_GRN_list.shape[0] == base_GRN_list.shape[1]:
            sub_base_GRN = base_GRN_list
        else:
            sub_base_GRN = Extract_base_GRN(base_GRN_list, node_feature.index)

        # sub_base_GRN = crop_network(node_feature, sub_base_GRN)
        base_GRN_indices_tensor, base_GRN_indices_values = adj2tensor(np.array(sub_base_GRN))
        assert base_GRN_indices_tensor.shape[1] == base_GRN_indices_values.shape[0], "base_GRN：边的数量与权重出现错误！"
        assert base_GRN_indices_tensor.max() < x_normalized.shape[0], "base_GRN：边的最大索引超出表达谱范围！"

    data = Data(x=x_normalized,
                edge_index=indices_tensor,
                edge_attr=values_tensor,
                base_GRN_edge_index=base_GRN_indices_tensor,
                base_GRN_edge_attr=base_GRN_indices_values,
                y=pd.DataFrame(node_feature).index)
    return data

def extract_number(s):
    # 尝试从字符串中提取数字，如果提取不到则返回 None
    try:
        return int(''.join(filter(str.isdigit, s)))
    except ValueError:
        return None

def sort_key(s):
    num = extract_number(s)
    # 如果能提取出数字，则返回数字，否则返回原字符串
    return (num if num is not None else s)

def read_simulation_h5_file_to_list(file_path):
    with h5py.File(file_path, 'r') as hdf_file:
        # 初始化一个字典来存储读取的数据
        data_dict = {}

        # 获取所有的组名称，并进行排序
        group_names = sorted(hdf_file.keys(), key=sort_key)

        # 遍历排序后的组（每个 dataset_name）
        for dataset_name in group_names:
            group = hdf_file[dataset_name]  # 访问组
            data_dict[dataset_name] = {}  # 初始化子字典

            network_key_list = ['network', 'KEGG', 'DoRothEA', 'TRRUST']
            for network_key in network_key_list:
                if network_key in group.keys():
                    network_raw = group[network_key][()]
                    network_data = pd.DataFrame({
                        'TF': [str(row[0], 'utf-8') if isinstance(row[0], bytes) else row[0] for row in network_raw],
                        'Gene': [str(row[1], 'utf-8') if isinstance(row[1], bytes) else row[1] for row in network_raw],
                    })
                    data_dict[dataset_name][network_key] = network_data

            gebe_id_list = [str(item, 'utf-8') if isinstance(item, bytes) else item for item in
                               group['gene_id'][()]]
            data_dict[dataset_name]['gene_id'] = gebe_id_list
            df_names = sorted(group.keys())
            network_key_list.append('gene_id')
            for df_name in df_names:
                if df_name not in network_key_list:
                    data = group[df_name][:]  # 读取 dataset 转为 numpy 数组
                    data_dict[dataset_name][df_name] = data  # 存储到字典
    return data_dict

def split_dict_key(key):
    parts = key.split('_')
    num = int(parts[-1])
    return num

def select_all_type_dict_num(my_dict):
    num_list = []
    for key in my_dict.keys():
        if '_' not in key:
            # 如果发现键不包含 '_', 则返回 None
            return None
        # 分割字符串，并提取'_'后面的部分
        parts = key.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            num = int(parts[-1])
            num_list.append(num)
        else:
            # 如果发现键包含 '_' 但不是有效数字，则返回 None
            return None
    return num_list

def Building_batch_from_h5(filename='/home/wcy/RegulationGPT/Data/Simulation/joint_represent_input.h5',
                           num=None, base_GRN=None, device=None, test=False, network_key='network', args=None):
    '''
    该函数用于将模拟数据集转为适合训练和测试的Data结构或者Batch结构，将所有的测序数据、邻接矩阵和其他信息放在一个文件中
    :param filename: 请输入经过联合表示学习后的h5文件
    :param num: 如果是Test，则提取某个数据集，否则提取前num个数据集
    :param device:
    :param test:
    :return:DATA, Batch结构
    '''
    if args is None:
        args = pathway_Config()

    if test:
        All_data_list = read_simulation_h5_file_to_list(filename)
        all_type_num_list = select_all_type_dict_num(All_data_list)
        keys = All_data_list.keys()
        if isinstance(num, numbers.Number):
            multiomics_dict = All_data_list[list(keys)[num]]
        elif isinstance(num, str):
            multiomics_dict = All_data_list[num]
        else:
            raise TypeError("num 必须是数字或字符串")

        assert 'mRNA' in multiomics_dict.keys(), "The input pickle file must contain the 'mRNA' item!"
        assert 'share' in multiomics_dict.keys(), "The input pickle file must contain the 'mRNA' item!"
        assert 'gene_id' in multiomics_dict.keys(), "The input pickle file must contain the 'feature_id' item!"

        adj_matrix = Extract_base_GRN(multiomics_dict[network_key], multiomics_dict['gene_id'])

        if base_GRN is None and args.simu:
            base_GRN_link = modify_matrix(multiomics_dict['mRNA'], adj_matrix, args.TP_radio, args.TN_radio, args.seed)
            # print(args.TP_radio)
            # auc_score = calculate_auc(adj_matrix, modified_net)
        else:
            if all_type_num_list:
                if isinstance(base_GRN, list):
                    base_GRN_link = copy.deepcopy(base_GRN[split_dict_key(list(keys)[num]) - 1])
                else:
                    base_GRN_link = base_GRN
            else:
                base_GRN_link = copy.deepcopy(base_GRN)
        data_net_exp = matrix2Data(adj_matrix=adj_matrix,
                                   node_feature=pd.DataFrame(multiomics_dict['share'],
                                                             index=multiomics_dict['gene_id']),
                                   Previous_data=None,
                                   base_GRN_list=base_GRN_link).to(device)
        data_net_exp = matrix2Data(adj_matrix=adj_matrix,
                                   node_feature=pd.DataFrame(multiomics_dict['mRNA'],
                                                             index=multiomics_dict['gene_id']),
                                   Previous_data=data_net_exp,
                                   base_GRN_list=base_GRN_link).to(device)
        non_mrna_keys = [key for key in multiomics_dict.keys() if
                         key not in ['mRNA', 'share', 'gene_id', 'network', 'DoRothEA', 'KEGG',
                                     'TRRUST'] and 'new' not in key]
        for key in non_mrna_keys:
            data_net_exp = matrix2Data(adj_matrix=adj_matrix,
                                       node_feature=pd.DataFrame(multiomics_dict[key],
                                                                 index=multiomics_dict['gene_id']),
                                       Previous_data=data_net_exp,
                                       base_GRN_list=base_GRN_link).to(device)
        return data_net_exp, 1
    else:
        DATA_list = []
        All_data_list = read_simulation_h5_file_to_list(filename)
        edge_percent = []
        if num is None:
            num = len(All_data_list)
        all_type_num_list = select_all_type_dict_num(All_data_list)
        mRNA_dimensions = []
        # 遍历大字典中的每个子字典
        filtered_dict = {}
        for key, value in All_data_list.items():
            # 检查子字典中是否包含'mRNA'这个key
            if 'mRNA' in value:
                mRNA_dimensions.append(value['mRNA'].shape[1])
            if 'ATAC' in value:
                mRNA_dimensions.append(value['ATAC'].shape[1])
            if 'spRNA' in value:
                mRNA_dimensions.append(value['spRNA'].shape[1])

        for key, value in All_data_list.items():
            if all(x in value for x in ['mRNA', 'ATAC', 'spRNA']):
                # 分别获取三个数据的第二维度（列数）
                mRNA_dim = value['mRNA'].shape[1]
                ATAC_dim = value['ATAC'].shape[1]
                spRNA_dim = value['spRNA'].shape[1]

            if mRNA_dim == np.max(mRNA_dimensions) and ATAC_dim == np.max(mRNA_dimensions) and spRNA_dim == np.max(mRNA_dimensions):
                filtered_dict[key] = value

        if np.min(mRNA_dimensions) != np.max(mRNA_dimensions):
            # 遍历字典，处理 mRNA 数据
            for key, value in All_data_list.items():
                value['mRNA'] = value['mRNA'][:, :50]
                value['ATAC'] = value['ATAC'][:, :50]
                value['spRNA'] = value['spRNA'][:, :50]

        with tqdm(total=num, ncols=100, desc="Processing datasets to DATA Batch!") as pbar:

            for idx, (dictname, multiomics_dict) in enumerate(All_data_list.items()):
                skip = 0    # Skip this data when there is a dimension mismatch
                if num is not None:
                    if num <= idx:
                        break
                assert 'mRNA' in multiomics_dict.keys(), "The input pickle file must contain the 'mRNA' item!"
                assert 'share' in multiomics_dict.keys(), "The input pickle file must contain the 'mRNA' item!"
                assert 'gene_id' in multiomics_dict.keys(), "The input pickle file must contain the 'feature_id' item!"
                assert 'network' in multiomics_dict.keys(), "The input pickle file must contain the 'network' item!"
                adj_matrix = Extract_base_GRN(multiomics_dict['network'], multiomics_dict['gene_id'])
                if base_GRN is None:
                    base_GRN_link = modify_matrix(multiomics_dict['mRNA'], adj_matrix, args.TP_radio, args.TN_radio,
                                                  args.seed)
                else:
                    if all_type_num_list:
                        base_GRN_link = copy.deepcopy(base_GRN[split_dict_key(dictname)-1])
                    else:
                        base_GRN_link = copy.deepcopy(base_GRN)

                data_net_exp = matrix2Data(adj_matrix=adj_matrix,
                                           node_feature=pd.DataFrame(multiomics_dict['share'],
                                                                     index=multiomics_dict['gene_id']),
                                           Previous_data=None,
                                           base_GRN_list=base_GRN_link).to(device)
                if multiomics_dict['share'].shape[1] < 50 or multiomics_dict['mRNA'].shape[1] < 50:
                    skip = 1
                data_net_exp = matrix2Data(adj_matrix=adj_matrix,
                                           node_feature=pd.DataFrame(multiomics_dict['mRNA'],
                                                                     index=multiomics_dict['gene_id']),
                                           Previous_data=data_net_exp,
                                           base_GRN_list=base_GRN_link).to(device)
                non_mrna_keys = [key for key in multiomics_dict.keys() if
                                 key not in ['mRNA', 'share', 'gene_id', 'network', 'DoRothEA', 'KEGG', 'TRRUST'] and 'new' not in key]
                for key in non_mrna_keys:
                    data_net_exp = matrix2Data(adj_matrix=adj_matrix,
                                               node_feature=pd.DataFrame(multiomics_dict[key],
                                                                         index=multiomics_dict['gene_id']),
                                               Previous_data=data_net_exp,
                                               base_GRN_list=base_GRN_link).to(device)
                    if multiomics_dict[key].shape[1] < 50:
                        skip = 1
                if skip == 0:
                    DATA_list.append(data_net_exp)
                    edge_percent.append(
                        np.sum(np.sum(adj_matrix)) / (multiomics_dict['mRNA'].shape[0] * multiomics_dict['mRNA'].shape[0] - multiomics_dict['mRNA'].shape[0]))
                pbar.update(1)

        edge_percent = sum(edge_percent) / len(edge_percent)
        batch = Batch.from_data_list(DATA_list)
        print("INFO:: Transformer datasets to DATA structure successfully!")
        return batch, edge_percent


if __name__ == '__main__':
    from multiomics_integration import Multiomics_learning
    # All_data = Multiomics_learning(ALL_data_file_path, rerun=False)
    test_filename = '/home/wcy/RegulationGPT/Data/PBMC_Healthy_Donor_Granulocytes_3k/cicero_output/joint_represent_pathway_train.h5'
    data, _ = Building_batch_from_h5(
        filename=test_filename,
        num=None,
        device='cuda',
        test=False)
