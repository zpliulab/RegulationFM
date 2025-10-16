import copy
import os
import pickle
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import mutual_info_score
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from diffusion_model.Building_training_dataset.PCA_CMI import pca_cmi
from diffusion_model.Building_training_dataset.PCA_PCC import pca_pcc
from diffusion_model.discrete.diffusion_utils import cal_identify_TF_gene
from functools import lru_cache
from config import pathway_Config
import h5py
from sklearn.neighbors import NearestNeighbors


# mapminmax
def MaxMinNormalization(x, Min=0, Max=1):
    x = (x - Min) / (Max - Min)
    return x


# add high MI
def high_MI(exp_pca_discretized, exp_pca, net_bit, parm):
    row_MI = compute_mutual_information(exp_pca_discretized)
    np.fill_diagonal(row_MI.to_numpy(), 0)
    # MI_thrd = 0.3
    expi = 1
    MI_thrd = (np.exp((expi * 0.01) ** 2) - 1)
    rflag = 1
    while rflag == 1:
        indices = np.where(row_MI > MI_thrd)
        # if parm['MI_percent'] * len(indices[0]) > net_bit.shape[0]:
        #     MI_thrd = MI_thrd + 0.1
        if len(indices[0]) > int(exp_pca.shape[0] * parm['MI_percent']):
            expi += 1
            MI_thrd = (np.exp((expi * 0.01) ** 2) - 1)
            rflag = 1
        else:
            MI_TF = exp_pca.index[indices[0]]
            MI_Gene = exp_pca.index[indices[1]]
            MI_TF_Gene = pd.DataFrame([MI_TF, MI_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, MI_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, MI_TF_Gene


# add high Pearson corrlation
def high_pearson(exp_pca, net_bit, parm):
    row_corr = exp_pca.T.corr(method='pearson')
    np.fill_diagonal(row_corr.to_numpy(), 0)
    # pearson_thrd = 0.95
    expi = 1
    pearson_thrd = (np.exp((expi * 0.01) ** 2) - 1)
    rflag = 1
    while rflag == 1:
        indices = np.where(row_corr > pearson_thrd)
        #   if parm['pear_percent'] * len(indices[0]) > net_bit.shape[0]:
        #    pearson_thrd = pearson_thrd + 0.0005
        if len(indices[0]) > int(exp_pca.shape[0] * parm['pear_percent']):
            expi += 1
            pearson_thrd = (np.exp((expi * 0.01) ** 2) - 1)
            rflag = 1
        else:
            corr_TF = exp_pca.index[indices[0]]
            corr_Gene = exp_pca.index[indices[1]]
            corr_TF_Gene = pd.DataFrame([corr_TF, corr_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, corr_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, corr_TF_Gene


def delte_low_corr(net_bit, node_feature):
    thr = 0.05
    correlation = node_feature.T.corr(method='pearson')
    # 如果链接关系表的数量超过节点数量的10倍
    while net_bit.shape[0] > 10 * node_feature.shape[0]:
        net_bit = net_bit[net_bit.apply(lambda edge: correlation.loc[edge[0], edge[1]] >= thr, axis=1)]
        thr += 0.05
    return net_bit


# calculate each type percent of edges in Gene_TF_list
def cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig):
    new_bit_crop = new_bit_crop['TF'] + '-' + new_bit_crop['Gene']
    if new_bit_crop.shape[0] == 0:
        return 0, 0, 0
    net_bit_origC = net_bit_orig['TF'] + '-' + net_bit_orig['Gene']
    net_bit_origC = pd.Series(list(set(new_bit_crop) & set(net_bit_origC)))
    NUM_ORIG = net_bit_origC.shape[0] / new_bit_crop.shape[0] * 100

    if len(corr_TF_Gene) > 0:
        corr_TF_GeneC = corr_TF_Gene['TF'] + '-' + corr_TF_Gene['Gene']
        corr_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(corr_TF_GeneC)))
        count_PCC = (~corr_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_PCC = count_PCC / new_bit_crop.shape[0] * 100
    else:
        NUM_PCC = 0

    if len(MI_TF_Gene) > 0:
        MI_TF_GeneC = MI_TF_Gene['TF'] + '-' + MI_TF_Gene['Gene']
        MI_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(MI_TF_GeneC)))
        count_MI = (~MI_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_MI = count_MI / new_bit_crop.shape[0] * 100
    else:
        NUM_MI = 0

    if (NUM_ORIG + NUM_PCC + NUM_MI) != 100:
        SUM1 = (NUM_PCC + NUM_MI)
        NUM_PCC = NUM_PCC * (100 - NUM_ORIG) / SUM1
        NUM_MI = NUM_MI * (100 - NUM_ORIG) / SUM1

    if NUM_PCC + NUM_MI > 50:
        overflow = True
    else:
        overflow = False
    return NUM_ORIG, NUM_PCC, NUM_MI, overflow

# cal MI
def compute_mutual_information(df):
    features = df.values
    num_rows = len(features)
    mi_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            mi = mutual_info_score(features[i], features[j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    mi_matrix = pd.DataFrame(mi_matrix, index=df.index, columns=df.index)
    return mi_matrix


def compare_char(charlist, setlist):
    try:
        index = setlist.index(charlist)
    except ValueError:
        index = None
    return index


def calRegnetwork(human_network, GRN_GENE_symbol):
    human_network = human_network[
        human_network['TF'].isin(GRN_GENE_symbol) & human_network['Gene'].isin(GRN_GENE_symbol)]
    human_network_TF_symbol = np.array(human_network['TF'])
    human_network_Gene_symbol = np.array(human_network['Gene'])

    if 'Score' in human_network.columns:
        human_network['Key'] = human_network['TF'] + '-' + human_network['Gene']
        Score_dict = pd.Series(human_network['Score'].values, index=human_network['Key']).to_dict()

    d = 1
    network = []

    for i in range(len(GRN_GENE_symbol)):
        number = [j for j, x in enumerate(human_network_TF_symbol) if str(GRN_GENE_symbol[i]) == x]
        if len(number) > 0:
            for z in range(len(number)):
                networkn = []
                number2 = compare_char(str(human_network_Gene_symbol[number[z]]), GRN_GENE_symbol)
                if number2 is not None:
                    networkn.append(GRN_GENE_symbol[i])  # 调控基因
                    networkn.append(GRN_GENE_symbol[number2])  # 靶基因
                    if 'Score' in human_network.columns:
                        networkn.append(
                            Score_dict[GRN_GENE_symbol[i] + '-' + GRN_GENE_symbol[number2]])  # Score for TF-Gene
                    network.append(networkn)
                    d += 1
    if 'Score' in human_network.columns:
        network = pd.DataFrame(network, columns=['TF', 'Gene', 'Score'])
    else:
        network = pd.DataFrame(network, columns=['TF', 'Gene'])

    return network


def load_KEGG(kegg_file='diffusion_model/Building_training_dataset/kegg/KEGG_all_pathway.pkl'):
    '''
        load kegg pathway
    '''
    if os.path.exists(kegg_file):
        # 如果 pkl 文件存在，则加载它
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    else:
        # 如果 pkl 文件不存在，则运行 KEGG.py 文件
        subprocess.call(['python', 'diffusion_model/Building_training_dataset/kegg/KEGG_process.py'])
        # 加载生成的 pkl 文件
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    return KEGG


def cal_metacell(BRCA_exp_filter_saver, Cnum=100, k=20):
    # 转置DataFrame以按列计算邻居
    BRCA_exp_filter_savert = BRCA_exp_filter_saver.transpose()
    # 使用KNN计算每列的前k个邻居
    neigh = NearestNeighbors(n_neighbors=k, metric='minkowski')
    neigh.fit(BRCA_exp_filter_savert)
    # 获取每列的前k个邻居的索引
    K_list = neigh.kneighbors(BRCA_exp_filter_savert, return_distance=False)
    ALL_C_list = list(range(BRCA_exp_filter_saver.shape[1]))
    max_consecutive_updates = Cnum*2
    S = [None for x in range(Cnum)]
    old_S = S.copy()
    Nc_max_list = np.zeros((1, Cnum))
    counter = 0
    if BRCA_exp_filter_savert.shape[0] <= Cnum:
        print(f"The number of cells to be processed ({str(BRCA_exp_filter_savert.shape[0])}) is less than (or equal) the number of Meta-cells ({str(Cnum)})!")
        return BRCA_exp_filter_saver
    while counter < max_consecutive_updates:
        ALL_C_list_current = [x for x in ALL_C_list if x not in S]
        for c in ALL_C_list_current:
            if c not in S:
                Nc_max = len(set(K_list[c]))
                for j in S:
                    if j is not None:
                        Nc = len(set(K_list[c]) | set(K_list[j]))
                        if Nc > Nc_max:
                            Nc_max = Nc
                if np.any(Nc_max > Nc_max_list):
                    S[np.argmin(Nc_max_list)] = c
                    Nc_max_list[0, np.argmin(Nc_max_list)] = Nc_max
                elif Nc_max == (k*2) and c < np.max(S):
                    S[np.argmax(S)] = c
                    Nc_max_list[0, np.argmax(S)] = Nc_max
                if np.array_equal(S, old_S):
                    counter += 1
                else:
                    old_S = S.copy()
                    counter = 0
        for cn in range(Cnum):
            c = S[cn]
            Nc_max = len(set(K_list[c]))
            for j in S:
                if j is not None and j!=c:
                    Nc = len(set(K_list[c]) | set(K_list[j]))
                    if Nc > Nc_max:
                        Nc_max = Nc
            if np.any(Nc_max > Nc_max_list[0, cn]):
                S[cn] = c
                Nc_max_list[0, cn] = Nc_max
    S = np.sort(S)
    assert None not in S, "Meta-cell list contains None!!!"
    assert len(S) == len(set(S)), "Meta-cell list contains duplicate values!!!"
    BRCA_exp_filter_saver = pd.DataFrame()
    for si in range(0, Cnum):
        new_value = (
            BRCA_exp_filter_savert.iloc[K_list[S[si], 0:int(BRCA_exp_filter_savert.shape[0] / Cnum)], :].mean(axis=0))
        BRCA_exp_filter_saver[str('c' + str(si))] = new_value

    return BRCA_exp_filter_saver


def Extract_construct_dataset_by_pathways(GE_Matrix, KEGG, parm, lim=300, exist_GRN=False, test=False,
                                          test_pathway=None, Other_Pathway=None,
                                          database_network=None, metacell=True, Cnum=200, k=20):
    if test_pathway is not None:
        exp_filter_NoBRCA = GE_Matrix.loc[~GE_Matrix.index.isin(KEGG[test_pathway])]
        exp = exp_filter_NoBRCA.loc[exp_filter_NoBRCA.index.isin(KEGG[Other_Pathway])]
    else:
        if Other_Pathway[:3] == "hsa":  # test belong to KEGG database
            exp = GE_Matrix.loc[GE_Matrix.index.isin(KEGG[Other_Pathway])]
        else:
            user_define = pd.read_csv(Other_Pathway)
            exp = GE_Matrix.loc[GE_Matrix.index.isin(user_define.iloc[:, -1].tolist())]
    if lim is None:
        lim = 300
    if (exp.shape[0] < 10 or exp.shape[0] > lim) and (Other_Pathway[:3] == "hsa"):
        return None, None, None
    if metacell:
        exp = cal_metacell(exp, Cnum=Cnum, k=k)
    if exist_GRN:
        return exp, None, None

    net_bit = calRegnetwork(database_network, exp.index.to_list())
    net_bit_orig = copy.deepcopy(net_bit)
    if test:
        nodes = np.unique(exp.index)
        adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
        for _, row in net_bit.iterrows():
            i = np.where(nodes == row['TF'])[0][0]
            j = np.where(nodes == row['Gene'])[0][0]
            adj_matrix[i, j] = 1

        return exp, adj_matrix, None

    # pro-process data
    exp_pca = exp
    exp_pca_discretized = pd.DataFrame()
    num_bins = 256
    for column in exp_pca.columns:
        bins = np.linspace(exp_pca[column].min(), exp_pca[column].max(), num_bins + 1)
        #      bins = exp_pca[column].quantile(q=np.linspace(0, 1, num_bins + 1))  # 根据分位数生成等频的区间
        labels = range(num_bins)
        if np.sum(exp_pca[column]) == 0:
            exp_pca_discretized[column] = exp_pca[column]
        else:
            exp_pca_discretized[column] = pd.cut(exp_pca[column], bins=bins, labels=labels, include_lowest=True)  # 执行离散化

    # add high link
    net_bit, corr_TF_Gene = high_pearson(exp_pca, net_bit, parm)
    net_bit, MI_TF_Gene = high_MI(exp_pca_discretized, exp_pca, net_bit, parm)
    # print(
    #     f'***********************      Reg: {net_bit_orig.shape[0]}           pearson: {corr_TF_Gene.shape[0]}              MI: {MI_TF_Gene.shape[0]}')

    if net_bit.shape[1] < 1:
        return None, None, None

    # creat adj
    nodes = np.unique(exp.index)
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for _, row in net_bit.iterrows():
        i = np.where(nodes == row['TF'])[0][0]
        j = np.where(nodes == row['Gene'])[0][0]
        adj_matrix[i, j] = 1

    GENE_ID_list, TF_ID_list = cal_identify_TF_gene(exp.index)
    for TF_ID in TF_ID_list:
        for GENE_ID in GENE_ID_list:
            adj_matrix[GENE_ID, TF_ID] = 0  # Gene -> TF is error

    predicted_adj_matrix, new_graph = pca_cmi(exp_pca_discretized, net_bit, parm['pmi_percent'], 1)
    predicted_adj_matrix = predicted_adj_matrix.toarray()
    new_bit_crop = pd.DataFrame(new_graph.edges(), columns=['TF', 'Gene'])
    CMI_SAVE_percent = new_bit_crop.shape[0]/net_bit.shape[0]
    if np.sum(predicted_adj_matrix) == 0:
        new_row = {'Pathway': Other_Pathway,
                   'NUM_ORIG': 0,
                   'NUM_PCC': 0,
                   'NUM_MI': 0,
                   'Links_ORIG': net_bit_orig.shape[0],
                   'CMI_SAVE_percent': CMI_SAVE_percent}
        return None, None, new_row
    elif (np.sum(adj_matrix) / np.sum(predicted_adj_matrix)) < 0.5:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(net_bit_orig, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'Pathway': Other_Pathway,
                   'NUM_ORIG': NUM_ORIG,
                   'NUM_PCC': NUM_PCC,
                   'NUM_MI': NUM_MI,
                   'Links_ORIG': net_bit_orig.shape[0],
                   'CMI_SAVE_percent': CMI_SAVE_percent}
        return exp, adj_matrix, new_row
    else:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'Pathway': Other_Pathway,
                   'NUM_ORIG': NUM_ORIG,
                   'NUM_PCC': NUM_PCC,
                   'NUM_MI': NUM_MI,
                   'Links_ORIG': net_bit_orig.shape[0],
                   'CMI_SAVE_percent': CMI_SAVE_percent}
        return exp, predicted_adj_matrix, new_row


def find_entrez_id(genome, genename):
    matching_row = genome[genome['symbol'] == genename]
    if not matching_row.empty:
        # 如果找到匹配的行，则输出对应的 'entrez_id'
        entrez_id = matching_row['entrez_id'].iloc[0]
        return entrez_id
    else:
        return -1

# 加载数据库知识
def return_database(database='RegNetwork', species='human', database_path='diffusion_model/Building_training_dataset'):
    if database == 'RegNetwork':
        # 加载RegNetwork数据库
        if species == 'human':
            Regnetwork_path = database_path+'/Regnetwork/2022.human.source'
        elif species == 'mouse':
            Regnetwork_path = database_path+'/Regnetwork/2022.mouse.source'
        else:
            print('Species error!')
        dtypes = {1: str, 3: str}
        human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
        human_network.columns = ['TF', 'TF_ID', 'Gene', 'Gene_ID']
    elif database == 'DoRothEA':
        if species == 'human':
            DoRothEA_path = database_path + '/DoRothEA/DoRothEA.csv'
            human_network = pd.read_csv(DoRothEA_path, sep=',', index_col=0)
            human_network.columns = ['TF', 'Gene', 'Repression']
        else:
            print('DoRothEA:: Species error!')

    elif database == 'KEGG':
        #  加载KEGG数据库
        if species == 'human':
            KEGG_path = database_path+'/kegg/KEGG_human_network_2022.csv'
        elif species == 'mouse':
            KEGG_path = database_path+'/kegg/KEGG_mouse_network_2022.csv'
        else:
            print('Species error!')
        human_network = pd.read_csv(KEGG_path, sep=',', index_col=0)
        human_network.columns = ['TF', 'TF_ID', 'Gene', 'Gene_ID']

    elif database == 'STRING':
        #  加载KEGG数据库
        if species == 'human':
            STRING_ID_path = database_path+'/STRING/9606.protein.aliases.v12.0.txt'
            STRING_Link_path = database_path+'/STRING/9606.protein.links.v12.0.txt'
            Genome_path = database_path+'/Genome.txt'
            dtypes = {0: str, 1: str}
            Genome_ID = pd.read_csv(Genome_path, sep='\t', dtype=str)

        elif species == 'mouse':
            STRING_ID_path = database_path+'/STRING/10090.protein.aliases.v12.0.txt'
            STRING_Link_path = database_path+'/STRING/10090.protein.links.v12.0.txt'
            Genome_path = database_path+'/MRK_List2.rpt'
            Genome_ID = pd.read_csv(Genome_path, sep='\t', dtype=str)
            Genome_ID = Genome_ID[Genome_ID['Marker Type'] == 'Gene']
            Genome_ID['symbol'] = Genome_ID['Marker Symbol']
        else:
            print('Species error!')

        protein_ID = pd.read_csv(STRING_ID_path, sep='\t', dtype=dtypes)
        protein_ID = protein_ID.loc[protein_ID['alias'].isin(Genome_ID['symbol'])]  # 只保留有重要意义的基因组
        protein_ID = protein_ID.drop_duplicates(subset=['#string_protein_id'])  # 去除重复值

        human_network = pd.read_csv(STRING_Link_path, sep=' ')
        human_network = human_network.loc[human_network['protein1'].isin(protein_ID['#string_protein_id'])]
        human_network = human_network.loc[human_network['protein2'].isin(protein_ID['#string_protein_id'])]

        # protein ENSP转symbol
        ENSP_to_gene = protein_ID.set_index('#string_protein_id')['alias'].to_dict()  # 制作一个查询字典
        human_network['protein1'] = human_network['protein1'].map(ENSP_to_gene)
        human_network['protein2'] = human_network['protein2'].map(ENSP_to_gene)
        human_network.columns = ['TF', 'Gene', 'Score']
        human_network = human_network.loc[human_network['Score'] > 800]

    elif database == 'TRRUST':
        #  加载TRRUST数据库
        if species == 'human':
            TRRUST_path = database_path+'/TRRUST/trrust_rawdata.human.tsv'
        elif species == 'mouse':
            TRRUST_path = database_path+'/TRRUST/trrust_rawdata.mouse.tsv'
        else:
            print('Species error!')
        human_network = pd.read_csv(TRRUST_path, sep='\t', header=None)
        human_network.columns = ['TF', 'Gene', 'Repression', 'id']

    else:
        raise ValueError('database is not supported')

    return human_network


def fill_nan2PCC(adj_matrix, node_feature):
    correlation_matrix = node_feature.T.corr(method='pearson')
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if np.isnan(adj_matrix[i, j]):
                adj_matrix[i, j] = correlation_matrix.iloc[i, j]
            if np.isnan(adj_matrix[i, j]):
                adj_matrix[i, j] = 0.0001
    return adj_matrix


# 计算PC-PMI, PC-PCC
def cal_pmi_ppc(node_feature, Regnetwork_adj_matrix, net_bit, per_theta, max_order=1, all_edge_thr=1, show=False):
    """
    node_feature: 基因表达数据
    Regnetwork_adj_matrix: 基因调控网络的邻接矩阵
    net_bit: 基因调控网络的边
    per_theta: PMI和PPC的阈值，当数值<1时，则表示为阈值，否则取百分比的边
    max_order: 最大阶数
    show: 是否显示输出信息
    """
    theta = {}
    theta['PMI'] = 1
    theta['PPC'] = 1
    PMIflag_while = True
    PPCflag_while = True
    Sum_adj = 1e4
    net_bit = delte_low_corr(net_bit, node_feature)  # 如果边太多，则预先删除一下
    Sum_adj_pmi = Sum_adj_ppc = 1e4
    expi = 2

    while expi <= 15 and Sum_adj > all_edge_thr * node_feature.shape[0]:
        theta['PPC'] = (np.exp((expi * 0.01) ** 2) - 1) * 10 if per_theta['PPC'] > 1 else per_theta['PPC']
        theta['PMI'] = (np.exp((expi * 0.01) ** 2) - 1) * 100 if per_theta['PMI'] > 1 else per_theta['PMI']

        if Sum_adj_pmi > all_edge_thr * node_feature.shape[0] or PMIflag_while:
            binary_adjMatrix_pmi, predicted_graph_pmi = pca_cmi(node_feature, net_bit, theta=theta['PMI'],
                                                                max_order=max_order, L=-1, show=show)
            binary_adjMatrix_pmi = binary_adjMatrix_pmi.toarray()
            binary_adjMatrix_pmi[binary_adjMatrix_pmi != 0] = 1
            Sum_adj_pmi = np.sum(np.multiply(Regnetwork_adj_matrix, binary_adjMatrix_pmi))
            PMIflag_while = not PMIflag_while

        if Sum_adj_ppc > all_edge_thr * node_feature.shape[0] or PPCflag_while:
            binary_adjMatrix_ppc, predicted_graph_ppc = pca_pcc(node_feature, net_bit, theta=theta['PPC'],
                                                                max_order=max_order, show=show)
            binary_adjMatrix_ppc = binary_adjMatrix_ppc.toarray()
            binary_adjMatrix_ppc[binary_adjMatrix_ppc != 0] = 1
            Sum_adj_ppc = np.sum(np.multiply(Regnetwork_adj_matrix, binary_adjMatrix_ppc))
            PPCflag_while = not PPCflag_while

        binary_adjMatrix_ppc_pmi = np.multiply(np.multiply(Regnetwork_adj_matrix, binary_adjMatrix_ppc),
                                               binary_adjMatrix_pmi)
        Sum_adj = np.sum(binary_adjMatrix_ppc_pmi)
        overlap_pcc_pmi = round(Sum_adj / np.sum(Regnetwork_adj_matrix), 2) * 100

        expi = expi + 1

        if Sum_adj_pmi <= all_edge_thr * node_feature.shape[0] and Sum_adj_ppc <= all_edge_thr * node_feature.shape[0]:
            break
        if Sum_adj_pmi <= all_edge_thr * node_feature.shape[0] and not PPCflag_while:
            break
        if Sum_adj_ppc <= all_edge_thr * node_feature.shape[0] * 1.5 and not PMIflag_while:
            break
        if overlap_pcc_pmi <= per_theta['PMI'] or overlap_pcc_pmi <= per_theta['PPC']:
            break

    # 计算邻接矩阵
    adj_matrix_with_pmi = nx.to_numpy_array(predicted_graph_pmi, weight='strength')
    adj_matrix_with_pmi = fill_nan2PCC(adj_matrix_with_pmi, node_feature) if np.isnan(
        adj_matrix_with_pmi).any() else adj_matrix_with_pmi  # 填补NAN值
    if np.max(adj_matrix_with_pmi) != 0:
        adj_matrix_with_pmi = adj_matrix_with_pmi / np.max(adj_matrix_with_pmi)  # 归一化
    adj_matrix_with_ppc = nx.to_numpy_array(predicted_graph_ppc, weight='strength')
    adj_matrix_with_ppc = fill_nan2PCC(adj_matrix_with_ppc, node_feature) if np.isnan(
        adj_matrix_with_ppc).any() else adj_matrix_with_ppc
    if np.max(adj_matrix_with_ppc) != 0:
        adj_matrix_with_ppc = adj_matrix_with_ppc / np.max(adj_matrix_with_ppc)

    adj_matrix_with_pmi = np.multiply(binary_adjMatrix_pmi, adj_matrix_with_pmi)
    adj_matrix_with_ppc = np.multiply(binary_adjMatrix_ppc, adj_matrix_with_ppc)

    assert np.all((binary_adjMatrix_pmi == 1) <= (adj_matrix_with_pmi != 0))
    assert np.all((binary_adjMatrix_ppc == 1) <= (adj_matrix_with_ppc != 0))
    adj_pmi_ppc_mean = (adj_matrix_with_pmi + adj_matrix_with_ppc) / 2
    assert not np.isnan(adj_pmi_ppc_mean).any()
    return binary_adjMatrix_pmi, binary_adjMatrix_ppc, adj_pmi_ppc_mean

@lru_cache(maxsize=None)
def cached_return_database(database, species, database_path):
    return return_database(database, species, database_path)


def load_database(database, species='human', database_path='diffusion_model/Building_training_dataset'):
    # database 分多种情况讨论，如果databse是一个list变量
    if isinstance(database, list):
        human_network_list = []
        for str_base in database:
            human_network_list.append(cached_return_database(database=str_base, species=species, database_path=database_path))
        filtered_dfs = [df[['TF', 'Gene']] for df in human_network_list if 'TF' in df and 'Gene' in df]
        human_network = pd.concat(filtered_dfs, axis=0)
    elif isinstance(database, str):
        human_network = return_database(database=database, species=species, database_path=database_path)
    else:
        raise ValueError('database is not supported')

    return human_network


def adjacency_to_link(adjacency_matrix, node_names):
    """
    将邻接矩阵转为TF-Target-Weight的形式。
    参数:
    - adjacency_matrix: numpy array, 表示邻接矩阵
    - node_names: list, 与邻接矩阵行和列对应的节点名称
    返回:
    - pd.DataFrame: 包含三列 'TF', 'Target', 'Weight'
    """

    tf_list = []
    target_list = []
    weight_list = []
    num_nodes = len(node_names)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adjacency_matrix[i, j]
            if weight != 0:  # 过滤掉权重为0的
                tf_list.append(node_names[i])
                target_list.append(node_names[j])
                weight_list.append(weight)

    # 创建一个DataFrame
    result_df = pd.DataFrame({
        'TF': tf_list,
        'Target': target_list,
        'weight': weight_list
    })

    return result_df

def read_h5_file_to_list(file_path):
    with h5py.File(file_path, 'r') as hdf_file:
        # 初始化一个字典来存储读取的数据
        data_dict = {}
        gebe_id_list = [str(item, 'utf-8') if isinstance(item, bytes) else item for item in
                           hdf_file['gene_id'][()]]
        data_dict['gene_id'] = gebe_id_list
        df_names = sorted(hdf_file.keys())
        for df_name in df_names:
            if df_name not in ['gene_id']:
                data = hdf_file[df_name][:]  # 读取 dataset 转为 numpy 数组
                data_dict[df_name] = data  # 存储到字典
    return data_dict


def save_dcit_to_h5ad(data_dict, save_path, file_label="standard_input.h5"):
    new_folder = os.path.join(save_path, file_label)
    with h5py.File(new_folder, 'w') as hdf_file:
        for dataset_name, df_dict in data_dict.items():
            group = hdf_file.create_group(dataset_name)
            for df_name, df in df_dict.items():
                if df.dtype.type is np.float32 or df.dtype.type is np.float64:
                    group.create_dataset(df_name, data=df)
                else:
                    group.create_dataset(df_name, data=np.array(df.astype('S')))

    print(f'INFO:: h5ad file saved at {new_folder}!')


def Segmentation_pathway(filename, test_pathway='hsa05224', test=False, filelabel="standard_input_pathway.h5", multi_test_label=False, args=None):
    if args is None:
        args = pathway_Config()
    # 1. 加载数据
    multiomics_dict = read_h5_file_to_list(filename+'/standard_input.h5')
    assert 'mRNA' in multiomics_dict.keys(), "The input pickle file must contain the 'mRNA' item!"
    assert 'gene_id' in multiomics_dict.keys(), "The input pickle file must contain the 'feature_id' item!"
    non_mrna_keys = [key for key in multiomics_dict.keys() if
                     key not in ['mRNA', 'gene_id']]

    # 2.加载KEGG数据库信息
    KEGG = load_KEGG(args.kegg_file)

    # # 3. 读取Regnetwork信息
    if multi_test_label:
        database_network = []
        for database_list in args.database_list:
            database_network_single = load_database(database=database_list, species='human', database_path=args.database_path)
            database_network.append(database_network_single)
    else:
        database_network = load_database(database=args.database_list, species='human', database_path=args.database_path)
    # 4. Create exp-net DATA
    # network_percent = pd.DataFrame(columns={'Pathway': None, 'NUM_ORIG': None, 'NUM_PCC': None, 'NUM_MI': None})
    print('INFO:: Segmentation pathway starting!')
    Pathway_seg_dict = {}
    if test:
        parm = {'pear_percent': args.high_pear_percent,
                'MI_percent': args.high_MI_percent,
                'pmi_percent': args.pmi_percent}  # 这里 1 表示 找到的高相关性的边不低于节点数量的  1 倍
        input_exp = pd.DataFrame(multiomics_dict['mRNA'], index=multiomics_dict['gene_id'])

        if multi_test_label:
            database_network_list = database_network[1:]
            database_network = database_network[0]
            database_list_other = args.database_list[1:]

        [exp, adj_matrix, _] = Extract_construct_dataset_by_pathways(input_exp, KEGG, parm,
                                                                     lim=args.pathway_lim,
                                                                     test_pathway=None,
                                                                     Other_Pathway=test_pathway,
                                                                     database_network=database_network,
                                                                     metacell=args.metacell,
                                                                     Cnum=args.metacell_num,
                                                                     k=args.metacell_k)
        link_network = adjacency_to_link(adj_matrix, exp.index)
        dict_key = test_pathway
        if '/' in test_pathway:
             dict_key = 'default'
        Pathway_seg_dict[dict_key] = {}
        Pathway_seg_dict[dict_key]['mRNA'] = np.array(exp)
        Pathway_seg_dict[dict_key]['gene_id'] = np.array(exp.index)
        Pathway_seg_dict[dict_key]['network'] = np.array(link_network)

        if multi_test_label:
            for database_network_single, database_list in zip(database_network_list, database_list_other):
                [_, adj_matrix, _] = Extract_construct_dataset_by_pathways(input_exp, KEGG, parm,
                                                                             lim=args.pathway_lim,
                                                                             test_pathway=None,
                                                                             Other_Pathway=test_pathway,
                                                                             database_network=database_network_single,
                                                                             metacell=args.metacell,
                                                                             Cnum=args.metacell_num,
                                                                             k=args.metacell_k)
                link_network = adjacency_to_link(adj_matrix, exp.index)
                Pathway_seg_dict[dict_key][database_list] = np.array(link_network)
        print((f" Pathway: {test_pathway}, total contain {exp.shape[0]} genes, and {np.sum(adj_matrix)} links!"))

        for key in non_mrna_keys:
            input_exp = pd.DataFrame(multiomics_dict[key], index=multiomics_dict['gene_id'])
            [exp, _, _] = Extract_construct_dataset_by_pathways(input_exp, KEGG, parm, exist_GRN=True,
                                                                         lim=args.pathway_lim,
                                                                         test_pathway=None,
                                                                         Other_Pathway=test_pathway,
                                                                         database_network=database_network,
                                                                         metacell=args.metacell,
                                                                         Cnum=args.metacell_num,
                                                                         k=args.metacell_k)

            Pathway_seg_dict[dict_key][key] = np.array(exp)
        # filelabel = filelabel.replace(".h5", "_test.h5")
    else:
        pathway_ID_list = KEGG.keys()
        pathway_ID_list = list(pathway_ID_list)
        parm = {'pear_percent': args.high_pear_percent,
                'MI_percent': args.high_MI_percent,
                'pmi_percent': args.pmi_percent}  # 这里 1 表示 找到的高相关性的边不低于节点数量的  1 倍

        pbar = tqdm(pathway_ID_list, ncols=150)
        for Other_Pathway in pbar:
            input_exp = pd.DataFrame(multiomics_dict['mRNA'], index=multiomics_dict['gene_id'])
            [exp, adj_matrix, description] = Extract_construct_dataset_by_pathways(input_exp, KEGG, parm,
                                                                         lim=args.pathway_lim,
                                                                         test_pathway=test_pathway,
                                                                         Other_Pathway=Other_Pathway,
                                                                         database_network=database_network,
                                                                         metacell=args.metacell,
                                                                         Cnum=args.metacell_num,
                                                                         k =args.metacell_k)

            if exp is not None:
                link_network = adjacency_to_link(adj_matrix, exp.index)
                Pathway_seg_dict[Other_Pathway] = {}
                Pathway_seg_dict[Other_Pathway]['mRNA'] = np.array(exp)
                Pathway_seg_dict[Other_Pathway]['gene_id'] = np.array(exp.index)
                Pathway_seg_dict[Other_Pathway]['network'] = np.array(link_network)
                pbar.set_description(
                    f" Pathway: {Other_Pathway}, "
                    f"Genes: {exp.shape[0]} genes, "
                    f"Links: {np.sum(adj_matrix)}, "
                    f"Database link: {description['Links_ORIG']}, "
                    f"PCA-CMI: {description['CMI_SAVE_percent']}.")

                for key in non_mrna_keys:
                    input_exp = pd.DataFrame(multiomics_dict[key], index=multiomics_dict['gene_id'])
                    [exp, _, _] = Extract_construct_dataset_by_pathways(input_exp, KEGG, parm, exist_GRN=True,
                                                                        lim=args.pathway_lim,
                                                                        test_pathway=test_pathway,
                                                                        Other_Pathway=Other_Pathway,
                                                                        database_network=database_network,
                                                                         metacell=args.metacell,
                                                                         Cnum=args.metacell_num,
                                                                         k=args.metacell_k)
                    Pathway_seg_dict[Other_Pathway][key] = np.array(exp)

    save_dcit_to_h5ad(Pathway_seg_dict, filename, filelabel)

if __name__ == '__main__':
    ALL_data_file_path = '/home/wcy/RegulationGPT/Data/PBMC'
    Segmentation_pathway(ALL_data_file_path, test_pathway='hsa05224', test=False)
