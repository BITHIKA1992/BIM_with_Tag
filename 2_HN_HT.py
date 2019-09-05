import community
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import random
import pickle
import time

from MIA_calculate import MIP_path_to_target
from MIA_calculate import Maximum_Influence_In_Arborescence
from MIA_calculate import activation_probability
from MIA_calculate import Expected_Influence

######################### Load  #################################

DG_prime = nx.read_gpickle( "DG_prime.gpickle")
#user_tag_matrix = np.load("user_tag_matrix.npy", user_tag_matrix)
with open('considered_community.pickle', 'rb') as handle:
    considered_community = pickle.load(handle)

with open('user_map.pickle', 'rb') as handle:
    user_map = pickle.load(handle)

with open('top_1000_tags_map.pickle', 'rb') as handle:
    top_1000_tags_map = pickle.load(handle)

with open('cost_per_user.pickle', 'rb') as handle:
    cost_per_user = pickle.load(handle)

with open('cost_per_mtag.pickle', 'rb') as handle:
    cost_per_mtag = pickle.load(handle)

with open('community_tag_count.pickle', 'rb') as handle:
    community_tag_count = pickle.load(handle)

with open('considered_tag_count_1.pickle', 'rb') as handle:
    considered_tag_count_1 = pickle.load(handle)

############################################################################################################
### Budget slection
budget = np.linspace(1000.0, 8000.0, 8)

EI_setting_budget = dict()
#EI_setting_budget['count_prob']
#EI_setting_budget['trivalency']
#EI_setting_budget['WeightedCascade']


result = []


### High Degree Node + High Count Tag as SeedNode and SeedTag
print("*****************************************************************************************************")
print("***********      BaseLine TBIM-IV HN+HT -- SeedSet and Seed Tag Selection ***************************")

Seed_Set = []
T_prime_Set = []
node_outdeg_list = sorted(list(DG_prime.out_degree(DG_prime.nodes())) , key=lambda x: x[1], reverse=True)

sorted_nodes= [node_outdeg_list[i][0] for i in range(len(node_outdeg_list))]
sorted_1000tags = [top_1000_tags_map[ considered_tag_count_1[i][0] ] for i in range(len(considered_tag_count_1))]



######### seedset and initial T_prime selection for every budget #############
for b in range(len(budget)):
    B_u = budget[b]/2.0
    B_t = budget[b]/2.0
    
    S_prime = []
    total_user_cost = 0.0
    for user in sorted_nodes:
        if total_user_cost + cost_per_user[user] <= B_u:
            total_user_cost = total_user_cost + cost_per_user[user]
            S_prime.append(user)
    
    T_prime = []
    total_tag_cost = 0.0
    for tag in sorted_1000tags:
        if total_tag_cost + cost_per_mtag[tag] <= B_t:
            total_tag_cost = total_tag_cost + cost_per_mtag[tag]
            T_prime.append(tag)
    
    Seed_Set.append(S_prime)
    T_prime_Set.append(T_prime)
    
  

 ########### Expected influence calculation with the seed and T_prime
print("*****************************************************************************************************")
print("***********      Influence with (1) count specific prob setting                **********************")
print("*****************************************************************************************************") 

EI_budget = np.zeros(len(budget))
for i in range(len(budget)):
    start_ts = time.time()
    print("Total Budget:", budget[i])
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
        
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]]['prob_weight_vec'][T_prime_Set[i]]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )
    
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    
    print("Number of edges:", DG_prime_instance.number_of_edges())
    
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, \
                                                           theta= 0.1, weight = 'tag_accum_prob_weight')
    
    EI_budget[i] = Expected_Influence(DG_prime_instance, Seed_Set[i], All_MIIA_DG,\
                   l=2, weight = 'tag_accum_prob_weight')
    
    end_ts = time.time()
    print("Expected Influence:", EI_budget[i])
    result.append([EI_budget[i], len(Seed_Set[i]), end_ts-start_ts])
    #EI_setting_budget['count_prob']['RN+RT'].append(EI_budget[i])
    
EI_setting_budget['count_prob'] = result
result = []



print("*****************************************************************************************************")
print("***********      Influence with (2) trivalency prob setting                **********************")
print("*****************************************************************************************************") 

EI_budget = np.zeros(len(budget))
for i in range(len(budget)):
    start_ts = time.time()
    print("Total Budget:", budget[i])
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
        
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]]['prob_weight_vec_tri'][T_prime_Set[i]]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )
    
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    
    print("Number of edges:", DG_prime_instance.number_of_edges())
    
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, \
                                                           theta= 0.1, weight = 'tag_accum_prob_weight')
    
    EI_budget[i] = Expected_Influence(DG_prime_instance, Seed_Set[i], All_MIIA_DG,\
                   l=2, weight = 'tag_accum_prob_weight')
    
    end_ts = time.time()
    print("Expected Influence:", EI_budget[i])
    result.append([EI_budget[i], len(Seed_Set[i]), end_ts-start_ts])
    #EI_setting_budget['trivalency']['RN+RT'].append(EI_budget[i])

EI_setting_budget['trivalency'] = result
result = []

print("*****************************************************************************************************")
print("***********      Influence with (3) weighted cascade prob setting              **********************")
print("*****************************************************************************************************") 

EI_budget = np.zeros(len(budget))
for i in range(len(budget)):
    start_ts = time.time()
    print("Total Budget:", budget[i])
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
        
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]]['prob_weight_vec_WC'][T_prime_Set[i]]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )
    
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    
    print("Number of edges:", DG_prime_instance.number_of_edges())
    
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, \
                                                           theta= 0.1, weight = 'tag_accum_prob_weight')
    
    EI_budget[i] = Expected_Influence(DG_prime_instance, Seed_Set[i], All_MIIA_DG,\
                   l=2, weight = 'tag_accum_prob_weight')
    
    end_ts = time.time()
    print("Expected Influence:", EI_budget[i])
    result.append([EI_budget[i], len(Seed_Set[i]), end_ts-start_ts])
    #EI_setting_budget['WeightedCascade']['RN+RT'].append(EI_budget[i])

EI_setting_budget['WeightedCascade'] = result
result = []

with open('result_2_HN_HT.pickle', 'wb') as handle:
    pickle.dump(EI_setting_budget, handle)
