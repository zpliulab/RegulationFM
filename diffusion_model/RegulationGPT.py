from torch_geometric.data import DataLoader
from .Building_training_dataset import Building_batch_from_h5
from tqdm import tqdm
from datetime import datetime
import torch
from diffusion_model.discrete_diffusion_model import Discrete_diffusion
from diffusion_model.discrete import Multimodel_Transformer, network_preprocess
from diffusion_model.discrete.diffusion_utils import Evaluation, cal_identify_TF_gene
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
import warnings
import pickle
from joblib import Parallel, delayed
import os

warnings.filterwarnings("ignore")


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.__dict__.update(config)

def is_csv_or_xlsx(file_name):
    return file_name.lower().endswith(('.csv', '.xlsx'))


class RegulationGPT:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.show = args.show
        # config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择适应的设备
        print(device)
        self.n_layers = args.num_layer  # Transfomer  layer
        self.n_head = args.num_head  # Transfomer  header
        self.MLP_base_node = args.num_MLP  # MLP
        self.GTM_base_node = args.num_GTM  # Transfomer  hidden

        self.batch_size = args.batch_size  # 批次大小
        self.num_epoch = args.num_epoch  # 训练次数
        self.lr = args.LR  # 学习率
        self.test_interval = args.test_interval  # 测试间隔
        self.save_interval = args.save_interval  # 检查点保存间隔
        self.max_nodes = args.max_nodes  # 网络最大的节点数量
        self.diffusion_timesteps = args.diffusion_timesteps  # 扩散模型的 time-step参数
        self.test_pathway = args.test_pathway  # 需要测试的Pathway（KEGG）
        self.lr_break = 1e-6
        self.ensemble = args.ensemble  # 集成数量
        self.rep_num = args.n_rep  # 测试重复次数
        self.n_job = args.n_job  # 并行数量
        self.muti_process = True if self.n_job > 1 else False

        self.save_dir = args.save_dir  # 结果文件存储路径
        self.save_label = args.save_label
        self.setup_paths()

        printlabel1 = 'The parameters are：          Transformer Layer        Transformer Head       # of node (MLP)     # of node (Transformer)      lr'
        print(printlabel1)
        printlabel2 = '           （default:2）            （default:4）         （default:32）         （default:16）         （default:1e-4）'
        print(printlabel2)
        printlabel3 = '                               ' + '                ' \
                      + str(self.n_layers) + '                        ' + str(self.n_head) + '                    ' \
                      + str(self.MLP_base_node) + '                    ' + str(
            self.GTM_base_node) + '                    ' + str(
            self.lr)
        print(printlabel3)

    def load_test_data_netowrk(self, test_filename, num=0, base_GRN_link=None, network_key='network', pathway_args=None):
        Multi_data, _ = Building_batch_from_h5(filename=test_filename,
                                               num=num,
                                               device=self.device,
                                               base_GRN=base_GRN_link,
                                               test=True,
                                               network_key=network_key,
                                               args=pathway_args)
        #  'network', 'DoRothEA', 'KEGG', 'TRRUST'
        # 提取黄金标准网络
        truelabel, node_mask, _ = network_preprocess.to_dense(Multi_data.x, Multi_data.edge_index,
                                                              Multi_data.edge_attr,
                                                              training=True, max_num_nodes=Multi_data.x.shape[0])
        truelabel_discrete = truelabel.mask(node_mask, collapse=True)
        truelabel_discrete = truelabel_discrete.E.squeeze(0)
        return Multi_data, truelabel_discrete

    def load_test_data(self, test_filename, num=0, base_GRN_link=None, network_key_list=['network', 'DoRothEA', 'KEGG', 'TRRUST'], pathway_args=None):
        truelabel_discrete_list = {}
        if network_key_list != 'network':
            for network_key in network_key_list:
                Multi_data, truelabel_discrete = self.load_test_data_netowrk(test_filename,
                                                                             num=num,
                                                                             base_GRN_link=base_GRN_link,
                                                                             network_key=network_key,
                                                                             pathway_args=pathway_args)
                truelabel_discrete_list[network_key] = truelabel_discrete
        else:
            Multi_data, truelabel_discrete_list = self.load_test_data_netowrk(test_filename,
                                                                         num=num,
                                                                         base_GRN_link=base_GRN_link,
                                                                         network_key=network_key_list,
                                                                         pathway_args=pathway_args)
        return Multi_data, truelabel_discrete_list

    def load_Multiomics_data(self, train_data_list, base_GRN_link=None,pathway_args=None):
        '''
          train_filename: 请输入训练数据文件名，如果是模拟数据，请输入包含list文件（每个list包含'net'和'exp',具体请参考模拟数据）的pickle保存的文件
          n_train ： 在模拟数据训练中使能，该参数为训练的网络数量, 例如 n_train = 200
          n_test ： 在模拟数据训练中使能，该参数为测试的网络数量，例如 n_test = [1000,1010]S
        '''
        split_path, split_filename = os.path.split(train_data_list)
        cache_filename = os.path.join(split_path,split_filename.replace('.h5', '.cache'))
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as file:
                Multi_train_data_dict = pickle.load(file)
                Multi_train_data = Multi_train_data_dict['Multi_train_data']
                edge_percent = Multi_train_data_dict['edge_percent']
        else:
            # 加载训练集数据
            Multi_train_data, edge_percent = Building_batch_from_h5(filename=train_data_list,
                                                                    num=self.args.num_train_datasets,
                                                                    device=self.device,
                                                                    base_GRN=base_GRN_link,
                                                                    test=False)
            with open(cache_filename, 'wb') as file:
                pickle.dump({'Multi_train_data': Multi_train_data, 'edge_percent': edge_percent}, file)
        # 如果验证集数量大于0，则加载验证集
        Multi_test_list = []
        if self.args.num_val_datasets is not None:
            if self.args.val_datasets_path is None:
                self.args.val_datasets_path = train_data_list
            for val_i in range(self.args.num_val_datasets[0], self.args.num_val_datasets[1]):
                Multi_data, _ = Building_batch_from_h5(filename=self.args.val_datasets_path ,
                                                       num=val_i,
                                                       device=self.device,
                                                       base_GRN=base_GRN_link,
                                                       test=True,
                                                       args=pathway_args)
                # 提取黄金标准网络
                truelabel, node_mask, _ = network_preprocess.to_dense(Multi_data.x, Multi_data.edge_index,
                                                                      Multi_data.edge_attr,
                                                                      training=True, max_num_nodes=Multi_data.x.shape[0])
                truelabel_discrete = truelabel.mask(node_mask, collapse=True)
                truelabel_discrete = truelabel_discrete.E.squeeze(0)
                Multi_test_list.append({'testdata': Multi_data, 'truelabel_discrete': truelabel_discrete})

        # if edge_percent < 0.1:
        #     edge_percent = 0.1
        self.data = Multi_train_data
        self.edge_percent = edge_percent
        self.test_list = Multi_test_list
        self.graph_data_loader = DataLoader(Multi_train_data, batch_size=self.batch_size, shuffle=False)


    def load_pre_model(self, diffusion_pre):
        n_layers = diffusion_pre['n_layers']
        input_dims = diffusion_pre['input_dims']
        hidden_mlp_dims = diffusion_pre['hidden_mlp_dims']
        hidden_dims = diffusion_pre['hidden_dims']
        output_dims = diffusion_pre['output_dims']
        model = Multimodel_Transformer(n_layers=n_layers,
                                       input_dims=input_dims,
                                       hidden_mlp_dims=hidden_mlp_dims,
                                       hidden_dims=hidden_dims,
                                       output_dims=output_dims,
                                       act_fn_in=nn.LeakyReLU(),
                                       act_fn_out=nn.LeakyReLU())
        model = model.to(self.device)
        model.load_state_dict(diffusion_pre['model_state_dict'])
        diffusion = Discrete_diffusion(model,
                                       device=self.device,
                                       max_num_nodes=diffusion_pre['max_nodes'],
                                       timesteps=self.diffusion_timesteps,
                                       edge_percent=diffusion_pre['edge_percent'],
                                       net_key_par=self.args.net_key_par)
        diffusion = diffusion.to(self.device)
        return diffusion

    def setup_model(self):
        if self.max_nodes is None:
            self.max_nodes = network_preprocess.get_max_node(self.data)
        print(f'The maximum number of genes in the GRN is  {self.max_nodes}! ')

        train_node = self.data.batch.unique_consecutive(return_counts=True)[1].max().item()
        assert train_node <= self.max_nodes, 'The number of nodes in the training set exceeds the preset value!'

        # create network
        x_data_shapes = [value.shape[1] for key, value in self.data.items() if key.startswith('x')]

        self.hidden_mlp_dims = {'X': self.MLP_base_node * 4,
                                'E': self.MLP_base_node * 2,
                                'y': self.MLP_base_node * 2}
        self.hidden_dims = {'dx': self.MLP_base_node * 4,
                            'de': self.MLP_base_node * 2,
                            'dy': self.MLP_base_node * 2,
                            'n_head': self.n_head,
                            'dim_ffX': self.GTM_base_node * 4, 'dim_ffE': self.GTM_base_node * 2,
                            'dim_ffy': self.GTM_base_node * 2}
        self.input_dims = {'X': x_data_shapes,
                           'E': 1,
                           'y': 1}
        self.output_dims = {'X': x_data_shapes,
                            'E': 2,
                            'y': 1}
        self.model = Multimodel_Transformer(n_layers=self.n_layers,
                                            input_dims=self.input_dims,
                                            hidden_mlp_dims=self.hidden_mlp_dims,
                                            hidden_dims=self.hidden_dims,
                                            output_dims=self.output_dims,
                                            act_fn_in=nn.ReLU(),
                                            act_fn_out=nn.ReLU())

        self.diffusion = Discrete_diffusion(self.model, device=self.device, max_num_nodes=self.max_nodes,
                                            timesteps=self.diffusion_timesteps,
                                            edge_percent=self.edge_percent,
                                            net_key_par=self.args.net_key_par)
        self.model = self.model.to(self.device)
        self.diffusion = self.diffusion.to(self.device)
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.lr, weight_decay=1e-2)  # 定义优化器
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.95)

    def cal_final_net(self, data, drop_TF=True):
        data_final = []
        for ii in range(len(data)):
            data1 = data[ii]
            data_final.append(data1[-1])

        data = pd.DataFrame()
        for df in data_final:
            data = data.add(df, fill_value=0)
        if drop_TF:
            GENE_ID_list, TF_ID_list = cal_identify_TF_gene(data.index)
            for TF_ID in TF_ID_list:
                for GENE_ID in GENE_ID_list:
                    data.iloc[GENE_ID, TF_ID] = 0

        return data


    def train(self, train_data, base_GRN_link=None, pathway_args=None):
        '''
          train_filename: 请输入训练数据文件名，如果是模拟数据，请输入包含list文件（每个list包含'net'和'exp',具体请参考模拟数据）的pickle保存的文件
          n_train ： 在模拟数据训练中使能，该参数为训练的网络数量, 例如 n_train = 200
          n_test ： 在模拟数据训练中使能，该参数为测试的网络数量，例如 n_test = [1000,1010]
        '''

        self.load_Multiomics_data(train_data, base_GRN_link, pathway_args=pathway_args)
        self.setup_model()

        # epoch遍历
        print('Start training...')
        pbar = tqdm(range(self.num_epoch), ncols=100)
        self.best_mean_AUC = 0
        self.best_mean_AUC_file = []
        for epoch in pbar:
            self.diffusion.train()
            self.model.train()
            total_loss = []
            total_AUC = []
            for idx, batch_x in enumerate(self. graph_data_loader):
                loss, Train_AUC = self.diffusion(batch_x)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)  # 梯度clip，保持稳定性
                self.optimizer.step()
                total_loss.append(loss.item())
                total_AUC.append(Train_AUC)
                pbar.set_description(f"Epoch: {epoch + 1:4.0f} - Train_AUC: {Train_AUC:.4f} "
                                     f"- loss: {loss.item():.4f} ")
            if self.optimizer.param_groups[0]['lr'] > self.lr_break:
                self.scheduler.step()
            total_loss = np.mean(total_loss)
            total_AUC = np.mean(total_AUC)

            if (epoch + 1) % 10 == 0:
                print("  --- Epoch %d average Loss: %.4f mean Train AUC: %.4f  lr: %0.6f" % (
                    epoch + 1, total_loss, total_AUC, self.optimizer.param_groups[0]['lr']))

            # 每隔 save_interval 个 epoch 保存模型
            if (epoch + 1) % self.save_interval == 0:
                filename = f'checkpoint_{self.timestamp}_epoch{epoch + 1}.pth'
                filename = os.path.join(self.checkpoint_dir, filename)
                self.save_model(filename)

            # 每隔 test_interval 个 epoch 验证模型
            if ((epoch + 1) % self.test_interval == 0) and (epoch + 1) > 1:
                if self.test_list:
                    self.validation(epoch)

        if self.save_dir is not None:
            final_filename = f"{self.save_label}_RegulationGPT_{self.timestamp}.pth"
            self.save_model(os.path.join(self.save_dir, final_filename))

        if not self.best_mean_AUC_file:
            self.best_mean_AUC_file = filename


        if not hasattr(self, 'printf'):
            self.printf = None

        return self.best_mean_AUC, self.best_mean_AUC_file, self.printf

    def test(self, diffusion_pre, testdata, truelabel=None, return_all=False):
        self.diffusion = self.load_pre_model(diffusion_pre)
        if diffusion_pre['max_nodes'] < testdata.x.shape[0]:
            print("INFO:: The number of genes exceeds the limit of the pre-trained model!")
            return None
        self.diffusion.eval()
        all_adj_list = []
        if self.muti_process:
            print(f'Open multi process!')
            all_adj_list = Parallel(n_jobs=self.n_job)(
                delayed(self.evaluate)(testdata, truelabel) for i in range(self.ensemble))
        else:
            for j in range(0, self.ensemble):
                _, all_adj = self.diffusion.test_step(testdata, truelabel, show=self.show)
                all_adj_list.append(all_adj)
        adj_final = self.cal_final_net(all_adj_list)
        if return_all:
            return all_adj_list
        else:
            return adj_final

    def evaluate(self, testdata, truelabel):
        x_pred, all_adj = self.diffusion.test_step(testdata, truelabel, show=self.show)
        return all_adj

    def setup_paths(self):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = os.path.join('results_checkpoint', self.args.save_label)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_model(self, filename):
        self.checkpoint = {
            'max_nodes': self.max_nodes,
            'n_layers': self.n_layers,
            'input_dims': self.input_dims,
            'hidden_mlp_dims': self.hidden_mlp_dims,
            'hidden_dims': self.hidden_dims,
            'output_dims': self.output_dims,
            'model_state_dict': self.model.state_dict(),
            'edge_percent': self.edge_percent,
        }
        torch.save(self.checkpoint, filename)
        print('saved model at ' + filename)

    def validation(self, epoch):
        aucs = []
        results = {'AUC': [], 'AUPR': [], 'EPR': [], 'EP': [], 'F1': [], 'nodenum': []}
        self.diffusion.eval()  # 设置模型为评估模式（如果有Dropout或BatchNorm层）
        self.model.eval()
        for rep in range(0, self.rep_num):
            for testi in range(0, len(self.test_list)):
                test_listone = self.test_list[testi]
                testdata = test_listone['testdata']
                truelabel_discrete = test_listone['truelabel_discrete']
                with torch.no_grad():
                    all_adj_list = []
                    if self.muti_process:
                        all_adj_list = Parallel(n_jobs=self.n_job)(
                            delayed(self.evaluate)(testdata, truelabel_discrete) for i in range(self.ensemble))
                    else:
                        for i in range(0, self.ensemble):
                            x_pred, all_adj = self.diffusion.test_step(testdata, truelabel_discrete, show=self.show)
                            # performance = Evaluation(y_pred=x_pred.flatten(),
                            #                          y_true=truelabel_discrete.flatten())
                            all_adj_list.append(all_adj)

                    adj_final = self.cal_final_net(all_adj_list, drop_TF=True)
                    performance = Evaluation(y_pred=adj_final.to_numpy().flatten(),
                                             y_true=truelabel_discrete.flatten())
                    print(np.count_nonzero(truelabel_discrete.cpu().numpy()), ' - ', np.count_nonzero(adj_final.to_numpy()))
                    aucs.append(performance['AUC'])
                    print("  --- Epoch  %.4f Test AUC: %.4f  AUPR: %.4f EP: %.4f EPR: %.4f F1: %.4f" % (
                        epoch + 1, performance['AUC'], performance['AUPR'], performance['Ep'],
                        performance['Epr'], performance['F1']))
                results['AUC'].append(performance['AUC'])
                results['AUPR'].append(performance['AUPR'])
                results['EP'].append(performance['Ep'])
                results['EPR'].append(performance['Epr'])
                results['F1'].append(performance['F1'])
        printf = "  --- Epoch  %.4f Final+std： Test AUC: %.4f+%.4f  AUPR: %.4f+%.4f EP: %.4f+%.4f  EPR: %.4f+%.4f F1: %.4f+%.4f" % (
            epoch + 1,
            np.mean(results['AUC']), np.std(results['AUC']),
            np.mean(results['AUPR']), np.std(results['AUPR']),
            np.mean(results['EP']), np.std(results['EP']),
            np.mean(results['EPR']), np.std(results['EPR']),
            np.mean(results['F1']), np.std(results['F1']))
        print(printf)

        if np.mean(results['AUC'][-self.rep_num:]) > self.best_mean_AUC:
            self.best_mean_AUC = np.mean(results['AUC'])
            self.best_mean_AUC_file = self.checkpoint_dir + f'/checkpoint_{self.timestamp}_epoch{epoch + 1}.pth'
            print(f"  --- Best AUC is {self.best_mean_AUC}")
            print(f"  --- Best best_AUC_file is {self.best_mean_AUC_file}")
            self.printf = printf

    def evaluation(self, adj_final, truelabel, show=False, flag=False):
        performance = Evaluation(y_pred=np.array(adj_final), y_true=truelabel, flag=flag)
        if show:
            print(
                f"RegulationGPT*-AUROC: {performance['AUC']:.4f} "
                f"AUPRC: {performance['AUPR']:.4f} "
                f"Epr: {performance['Epr']:.4f} "
                f"F1-score: {performance['F1']:.4f}")
        return performance
