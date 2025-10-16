import numpy as np
import warnings
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
import warnings
from scipy.stats import ttest_1samp


def _stats2LinkList(tg, stat_df):

    links = pd.DataFrame({"source": stat_df.index.values,
                          "target": np.repeat(tg, len(stat_df))})
    linkList = pd.concat([links, stat_df.reset_index(drop=True)], axis=1)

    return linkList

def _get_stats_df_bagging_ridge(df):

    if isinstance(df, int):
        return 0

    mean = df.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = df.apply(lambda x: ttest_1samp(x.dropna(), 0)[1])
    neg_log_p = -np.log10(p.fillna(1))

    result = pd.concat([mean, mean.abs(),
                        p, neg_log_p, #positive_score, negative_score
                        ], axis=1, sort=False)
    result.columns = ["coef_mean", "coef_abs", "p", "-logp",
                      #"positive_score", "negative_score"
                      ]

    return result

def intersect(list1, list2):
    """
    Intersect two list and get components that exists in both list.

    Args:
        list1 (list): input list.
        list2 (list): input list.

    Returns:
        list: intersected list.

    """
    inter_list = list(set(list1).intersection(list2))
    return(inter_list)

def _get_coef_matrix(ensemble_model, feature_names):
    # ensemble_model: trained ensemble model. e.g. BaggingRegressor
    # feature_names: list or numpy array of feature names. e.g. feature_names=X_train.columns
    feature_names = np.array(feature_names)
    n_estimater = len(ensemble_model.estimators_features_)
    coef_list = \
        [pd.Series(ensemble_model.estimators_[i].coef_,
                   index=feature_names[ensemble_model.estimators_features_[i]])\
         for i in range(n_estimater)]

    coef_df = pd.concat(coef_list, axis=1, sort=False).transpose()

    return coef_df


def get_bagging_ridge_coefs(target_gene, gem_scaled, TFdict,
                 bagging_number=1000, n_jobs=-1, alpha=1, solver="auto"):
    ## 1. Data prep
    if target_gene not in TFdict.keys():
        #print("err")
        return 0

    # define regGenes
    reggenes = TFdict[target_gene]
    allgenes_detected = list(gem_scaled.columns)
    reggenes = intersect(reggenes, allgenes_detected)

    if target_gene in reggenes:
        reggenes.remove(target_gene)

    reg_all = reggenes.copy()

    if not reggenes: # if reqgene is empty, return 0
        return 0

    # prepare learning data
    data = gem_scaled[reg_all]
    label = gem_scaled[target_gene]
    model = BaggingRegressor(estimator=Ridge(alpha=alpha,
                                             solver=solver,
                                             random_state=2024),
                             n_estimators=bagging_number,
                             bootstrap=True,
                             max_features=0.8,
                             n_jobs=n_jobs,
                             verbose=False,
                             random_state=2024)
    model.fit(data, label)
    # get results
    coefs = _get_coef_matrix(model, reg_all)
    return coefs


class Bagging_ridge_net():

    def __init__(self):
        self.failed_genes = []
        self.stats_dict = {}
        self.fitted_genes = []
        self.bagging_number = 20
        self.alpha = 10
        self.RIDGE_SOLVER = 'auto'
        self.n_jobs = 20

    def createTFdict(self, base_GRN):
        tmp = base_GRN.copy()
        tmp = tmp.groupby(by="gene_short_name").sum()
        self.TFdict = dict(tmp.apply(lambda x: x[x > 0].index.values, axis=1))


    def select_2diff_node(self, linkList, ponit=None):
        linkList = linkList.sort_values('-logp', ascending=False)
        # diff = np.diff(linkList['-logp'])
        # second_derivative = np.diff(diff)
        # ponit = np.argmax(second_derivative[1:]) + 3
        linkList = linkList[linkList['p'] <= 0.1]
        if ponit is None:
            p_cumsum = np.cumsum(linkList['-logp']) / np.sum(linkList['-logp'])
            ponit = np.argmax(p_cumsum > 0.9)
        linkList = linkList.iloc[:ponit]
        return linkList

    def run_bagging_ridge(self, mRNA, genelist,base_GRN, ponit=None):
        # genelist = []  # 所有的基因列表
        # mRNA = []  # 基因表达谱  cell * gene
        base_GRN = pd.DataFrame(base_GRN, index=genelist, columns=genelist)
        base_GRN['gene_short_name'] = genelist
        self.createTFdict(base_GRN)
        for target_gene in genelist:
            coefs = get_bagging_ridge_coefs(target_gene=target_gene,
                                            TFdict=self.TFdict,
                                            gem_scaled=mRNA,
                                            bagging_number=self.bagging_number,
                                            n_jobs=self.n_jobs,
                                            alpha=self.alpha,
                                            solver=self.RIDGE_SOLVER)
            if isinstance(coefs, int):
                self.failed_genes.append(target_gene)
            else:
                self.fitted_genes.append(target_gene)
                self.stats_dict[target_gene] = _get_stats_df_bagging_ridge(coefs)
        linkList = self.updateLinkList()
        return self.select_2diff_node(linkList, ponit)

    def updateLinkList(self):
        """
        Update LinkList.
        LinkList is a data frame that store information about inferred GRNs.

        Args:
            verbose (bool): Whether or not to show a progress bar

        """
        if not self.fitted_genes: # if the sequence is empty
            print("No model found. Do fit first.")
        linkList = []
        loop = np.unique(self.fitted_genes)
        for i in loop:
            linkList.append(_stats2LinkList(i, stat_df=self.stats_dict[i]))
        linkList = pd.concat(linkList, axis=0)
        linkList = linkList.reset_index(drop=True)
        return linkList