import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

MAIN_DIR = Path('.').absolute().parent
sys.path.append(str(MAIN_DIR))
from dynosarc.dynamic import utils as utl
from dynosarc.dynamic import dynorc

###########
# Directories and file names 
###########

DATA_DIR = MAIN_DIR / 'data'
RESULTS_DIR = MAIN_DIR / 'results'

SUBTYPES_FNAME = DATA_DIR / 'subtypes.csv'  

RNA_HPRD_FNAME = DATA_DIR / 'rna_hprd.csv'  
E_HPRD_ONCOKB_FNAME = DATA_DIR / 'edgelist_hprd_oncokb.csv' 

CLUS_RESULTS_DIR = RESULTS_DIR / 'clus_results'
invr_fname = CLUS_RESULTS_DIR / 'invariant_distributions_hprd.csv'
emds_fname = CLUS_RESULTS_DIR / 'emds_invariant_distributions_hprd.csv'
clustering_fname = CLUS_RESULTS_DIR / 'clustering_invariant_distributions_hprd.csv'

DYNO_RESULTS_DIR = RESULTS_DIR / 'dyno_results'

if not DYNO_RESULTS_DIR.is_dir():
    DYNO_RESULTS_DIR.mkdir()    
    
#########
# Load data
#########

# true subtypes:
subtypes = pd.read_csv(SUBTYPES_FNAME, header=0, index_col=0)
subtypes = subtypes['subtype']

# Wasserstein-based clustering predicted subtypes:
wass_subtypes = pd.read_csv(clustering_fname, header=0, index_col=0)
wass_subtypes = wass_subtypes['Wasserstein_hclust']

subtypes = subtypes.loc[wass_subtypes.index]

# lut = {'DSRCT': 'darkblue', 'Ewing Sarcoma': 'darkgreen', 'Osteosarcoma': 'orange', 'Embryonal RMS': 'purple'}
# subtype_colors = subtypes.replace(lut).to_frame(name='Subtype')

# graph:
E_onco = pd.read_csv(E_HPRD_ONCOKB_FNAME, header=0, index_col=None)
G_onco = nx.from_pandas_edgelist(E_onco)
# print(G_onco)

E_HPRD_FNAME = DATA_DIR / 'edgelist_hprd.csv' 
E_hprd = pd.read_csv(E_HPRD_FNAME, header=0, index_col=None)
G_hprd = nx.from_pandas_edgelist(E_hprd)
# print(G_hprd)

# RNA-Seq expression profiles:
data = pd.read_csv(RNA_HPRD_FNAME, header=0, index_col=0)
data = data[subtypes.index.copy()]

##########
# Helper functions for dynamic analysis
##########

def subtype_ids(subtypes, subtype):
    """ Return ids of samples in specified subtype
    
    Parameters
    -----------
    subtypes : Pandas Series
        Subtype classification for each sample.
    subtype : {'EWS', 'OST', 'DSRCT', 'ERMS'}
        Specify subtype.
        
    Return
    ------
    sids : list
        List of ids of samples in specified subtype.
    """        
        
    if subtype in ['EWS', 'EWS_cluster']:
        subtype = 'Ewing Sarcoma'
    elif subtype in ['OST', 'OST_cluster']:
        subtype = 'Osteosarcoma'
    elif subtype == 'ERMS':
        subtype = 'Embryonal RMS'
    elif subtype == 'DSRCT':
        pass
    else:
        raise ValueError("Unrecognized subtype, must be one of ['EWS', 'OST', 'DSRCT', 'ERMS', 'EWS_cluster', 'OST_cluster'].")
            
    sids = subtypes.loc[subtypes==subtype].index.tolist()    
    return sids

def subtype_data(subtypes, subtype, graph='onco', return_copy=True):
    """ Return RNA expression profiles for samples in the specified subtype 
    
    Parameters
    -----------
    subtypes : Pandas Series
        Subtype classification for each sample.
    subtype : {'EWS', 'OST', 'DSRCT', 'ERMS'}
        Specify subtype.
    graph : {'onco', 'hprd'}
        Graph topology.
    return_copy : bool
        If True, return copy of original data.
        
    Return
    ------
    data : Pandas DataFrame
        Subset of data for samples in specified cohort.
        """

    samples = subtype_ids(subtypes, subtype)
    
    if graph == 'onco':
        gl = list(G_onco)
    elif graph == 'hprd':
        gl = list(G_hprd)
    else:
        raise ValueError("Unrecognized graph, must be one of ['onco', 'hprd'].")
        
    if return_copy:
        return data.loc[gl, samples].copy()
    else:
        return data.loc[gl, samples]
    
def weighted_network(subtypes, subtype, graph='onco', **corr_kws):
    """ return subtype specific weighted network 
    
    Parameters
    -----------
    subtypes : Pandas Series
        Subtype classification for each sample.
    subtype : {'EWS', 'OST', 'DSRCT', 'ERMS'}
        Specify subtype.
    graph : {'onco', 'hprd'}
        Graph topology.
    corr_kws : dict
        Keyword arguments for computed weighted network.
        
    Returns
    -------
    s_weights : Pandas DataFrame
        Dataframe with 3 columns including source, target, weight of the network (where weights correspond to distance).
    """
    s_data = subtype_data(subtypes, subtype, graph=graph)
    
    if graph == 'onco':
        edgelist = E_onco
    elif graph == 'hprd':
        edgelist = E_hprd
    else:
        raise ValueError("Unrecognized graph, must be one of ['onco', 'hprd'].")
            
    return utl.distance_weights(s_data, edgelist, **corr_kws)

file_ids = {"EWS": "ewing", "OST": "osteo", "DSRCT": "dsrct",
            # "EWS_cluster": "ewing", "OST_cluster": "osteo",
           }

def get_directory_prefix(subtype, graph='onco'):
    if graph == 'onco': 
        s = 'oncoKB'
        n = len(G_onco)        
    elif graph == 'hprd':
        s = 'HPRD'
        n = len(G_hprd)
    else:
        raise ValueError("Unrecognized graph, must be one of ['onco', 'hprd'].")
        
    return "_".join([subtype, s, str(n)])


def cohort_curvature_simulation(subtype, W, crit=0.75, graph='onco', directory=None, edgelist=None,
                                force_recompute=False, chunksize=None, **dyn_kws):
    """ load data and compute dynamic curvature analysis """
    
    if directory is None:
        directory = DYNO_RESULTS_DIR / get_directory_prefix(subtype, graph=graph) # + f"_corr_abs_value"
    G = nx.from_pandas_edgelist(W, edge_attr='weight')
    dyno = utl.run_curvature_simulation(str(directory), G, crit=crit, edgelist=edgelist,
                                        force_recompute=force_recompute, chunksize=chunksize, **dyn_kws)
    return dyno
        
            
def get_cohort_curvature_simulation_results(subtype, crit=0.75, graph='onco', directory=None,
                                            **dyn_kws):
    """ load DORC simulation """   
    if directory is None:
        directory = DYNO_RESULTS_DIR / get_directory_prefix(subtype, graph=graph)    
    dyno = utl.load_curvature_simulation(str(directory), crit=crit, **dyn_kws)
    return dyno    

#########
# Run HPRD-EWS dynamic simulation
#########

subtype_cur = 'EWS'
graph_cur = 'hprd'
chunksize = 5
data_cur = subtype_data(subtypes, subtype_cur, graph=graph_cur)
corr_kws_cur = {'method': 'pearson', 
                'min_samples': data_cur.shape[1] - 3,       
                'thresh': 0.03, 
                'std_thresh': 3}


W_fname = DYNO_RESULTS_DIR / "_".join([subtype_cur, graph_cur, 'weights.csv'])
if W_fname.is_file():
    print("Loading weights from file.")
    W_cur = pd.read_csv(W_fname, header=0)
else:
    print("Computing weights.")
    W_cur = weighted_network(subtypes, subtype_cur, graph=graph_cur, **corr_kws_cur)
    W_cur.to_csv(W_fname, header=True)
    print("Weights saved to file.")

dyn_kws = dict(times=None, 
               t_min=-1.8, # -2, # -2.2,
               t_max=0.85, # 0.833333333333333, # 2.0, 
               n_t=40, # 55,
               log_time=True,               
               use_spectral_gap=False,
               e_weight="weight",
               verbose="WARNING",
              )

edgelist_cur = list(G_onco.edges()) + [['EWSR1', 'ETV6'], ['EWSR1', 'WT1']]

# print(W_cur.head())
# print(f"Edgelist with {len(edgelist_cur)} edges.")
dyno_cur = cohort_curvature_simulation(subtype_cur, W_cur, graph=graph_cur, edgelist=edgelist_cur,
                                       crit=0.75,  force_recompute=False, chunksize=chunksize, **dyn_kws)

# dyno_cur = get_cohort_curvature_simulation_results(subtype_cur, graph='hprd',
#                                                    crit=0.75, **dyn_kws)




# CURVATURE STUFF
from dynosarc.dynamic import dynorc
W= pd.read_csv(DATA_DIR/'ews_network.csv')

G = nx.from_pandas_edgelist(W, 'node1','node2',edge_attr='weight')


# Create a mapping from string labels to integers
node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

# Create a new graph with integer node labels
G = nx.relabel_nodes(G, node_mapping)

# Print the mapping between string and integer labels
print("String to Integer Mapping:")
print(node_mapping)

# Print the edges of the new graph with integer labels
print("\nEdges of the New Graph with Integer Labels:")
print(G.edges())
# {'ACTG1': 0, 'ARID1A': 1, 'DCTN1': 2, 'PPP1CB': 3, 'AFF1': 4, 'MLLT1': 5, 'LASP1': 6, 'KDM6A': 7, 'ZNF384': 8, 'ETV6': 9, 'APH1A': 10, 'NCSTN': 11, 'STAG2': 12, 'MSH6': 13, 'U2AF1': 14, 'MAP2K4': 15, 'FUBP1': 16, 'TLE1': 17, 'FGFR2': 18, 'EWSR1': 19, 'BRCA2': 20, 'POLE': 21, 'SOX2': 22, 'BACH2': 23, 'PATZ1': 24, 'TRIP13': 25, 'NTHL1': 26, 'FANCE': 27, 'MAD2L2': 28, 'FANCD2': 29, 'FANCC': 30, 'RAD21': 31, 'PPM1D': 32, 'SMC1A': 33, 'FANCF': 34, 'LMNA': 35, 'XPA': 36, 'FANCG': 37, 'SPOP': 38, 'CYP19A1': 39, 'MUTYH': 40, 'ERCC5': 41, 'FANCA': 42, 'CACNA1D': 43, 'CASP8': 44, 'NOD1': 45, 'DDIT3': 46, 'MYD88': 47, 'CYLD': 48, 'CHIC2': 49, 'FGF10': 50, 'CLIP1': 51, 'NUMA1': 52, 'TRAF7': 53, 'IGF2': 54, 'RPL5': 55, 'DKK1': 56, 'POU5F1': 57, 'LRP6': 58, 'DKK2': 59, 'LRP5': 60, 'GPC3': 61, 'ZRSR2': 62, 'TLX1': 63, 'FLI1': 64, 'U2AF2': 65, 'PAX7': 66, 'TAF15': 67, 'NR4A3': 68, 'SIX1': 69, 'STAG1': 70, 'PRCC': 71, 'STK19': 72, 'RRAS': 73, 'PTPRD': 74, 'PTPRS': 75, 'RALGDS': 76, 'SUFU': 77}
#  EWSR1: 19
# FLI1: 64
#
# nx.get_edge_attributes(G,"weight")
# apply or curvature on weighted graph 
# dynamic diffusion process? 
# geodesic_distances = dynorc._compute_distance_geodesic(G,weight='weight')


# nx.set_edge_attributes(G, {(v0, v1): geodesic_distances[v0, v1].item() for v0, v1 in G.edges()}, name="distance")
# measures: list
# geodes dists : list
# measures = list(np.eye(len(G)))

# turn nodes into numbers instead of string names
# edge = ('EWSR1','FLI1')
edge = (19,64)
times = dynorc._get_times()

dynorc._compute_dynamic_curvatures(G,times=times,edgelist=[edge])