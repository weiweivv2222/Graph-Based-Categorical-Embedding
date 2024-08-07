from scipy.sparse import coo_matrix, csr_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split
from scipy import stats


def create_cooccurrence_matrix_diag(categories,df_cat):

    # Get unique categories

    categories = np.unique(df_cat)


    # Create a dictionary to map category to index

    category_to_index = {cat: i for i, cat in enumerate(categories)}


    matrix = np.zeros((len(categories), len(categories)))

    for row in df_cat.itertuples(index=False):

        for i, cat1 in enumerate(row):

            for j, cat2 in enumerate(row):

                if i != j:

                    matrix[category_to_index[cat1], category_to_index[cat2]] += 1



    # Calculate the diagonal values

    diagonal_values = []

    for cat in categories:

        count = df_cat.stack().value_counts()[cat]

        diagonal_values.append(count)



    # Calculate co-occurrence matrix

    matrix = np.zeros((len(categories), len(categories)))

    for row in df_cat.values:

        for i, cat1 in enumerate(row):

            for j, cat2 in enumerate(row):

                if i != j:

                    matrix[category_to_index[cat1], category_to_index[cat2]] += 1

    # Set diagonal values to category counts

    np.fill_diagonal(matrix, diagonal_values)



    # Convert the matrix to a sparse matrix

    sparse_matrix = coo_matrix(matrix)



    # Normalize the matrix

    normalized_matrix = csr_matrix(sparse_matrix / sparse_matrix.sum(axis=1))



    # Print the normalized matrix

    print(normalized_matrix.toarray())



    # Print the matrix with row and column labels

    co_occurrence_matrix_norm= pd.DataFrame(normalized_matrix.toarray(), index=categories, columns=categories)

    print(co_occurrence_matrix_norm)
    
    return co_occurrence_matrix_norm




import matplotlib.pyplot as plt


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857 
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    #ax.set_xticklabels(xticklabels, minor=False)
    #ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    #show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(40, 20))
    
    

    
    


    
def paired_ttest_5x2cv(estimator1, estimator2, X1,X2, y, scoring=None, random_seed=None):
    """
    Implements the 5x2cv paired t test proposed
    by Dieterrich (1998)
    to compare the performance of two models.
    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor
    estimator2 : scikit-learn classifier or regressor
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.
    Returns
    ----------
    t : float
        The t-statistic
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.
    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
    """
    rng = np.random.RandomState(random_seed)

    if scoring is None:
        if estimator1._estimator_type == "classifier":
            scoring = "accuracy"
        elif estimator1._estimator_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must " "be a Classifier or Regressor.")
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    variance_sum = 0.0
    first_diff = None

    def score_diff(X_11, X_12,X21,X22, y_1, y_2):
        estimator1.fit(X_11, y_1)
        estimator2.fit(X_21, y_1)
        est1_score = scorer(estimator1, X_12, y_2)
        est2_score = scorer(estimator2, X_22, y_2)
        score_diff = est1_score - est2_score
        return score_diff

    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        X_11, X_12, y_1, y_2 = train_test_split(X1, y, test_size=0.5, random_state=randint)
        X_21, X_22, y_1, y_2 = train_test_split(X2, y, test_size=0.5, random_state=randint)


        score_diff_1 = score_diff(X_11, X_12, X_21,X_22, y_1, y_2)
        score_diff_2 = score_diff(X_12, X_11, X_22, X_21, y_2, y_1)
        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1

    numerator = first_diff
    denominator = np.sqrt(1 / 5.0 * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return float(t_stat), float(pvalue)

    
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

#https://arxiv.org/pdf/1806.08804.pdf
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class DiffPool(torch.nn.Module):
    def __init__(self, num_features, num_classes, max_nodes=150):
        super().__init__()
        self.max_nodes = max_nodes
        num_nodes = num_classes #ceil(0.25 * max_nodes)
        
        self.gnn1_pass = GNN(num_features,64,num_features)
        
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = GNN(num_features, 64, 64, lin=False)

        num_nodes = ceil(0.5 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, max_nodes*max_nodes)
        #decode the adjacency matrix
        self.decode = torch.nn.Sigmoid()

    def forward(self, x, adj, mask=None):
        
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x1, adj1, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s1 = self.gnn2_pool(x1, adj1)
        x1 = self.gnn2_embed(x1, adj1)

        x2, adj2, l2, e2 = dense_diff_pool(x1, adj1, s1)

        x3 = self.gnn3_embed(x2, adj2)

        x3 = x3.mean(dim=1)
        x3 = self.lin1(x3).relu()
        x3 = self.lin2(x3)
        A = self.decode(x3)
        return A.reshape((self.max_nodes,self.max_nodes)),s,adj1,s1,adj2,l1+l2 #F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
    
    
    
