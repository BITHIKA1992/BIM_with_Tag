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

###### considered_community #######
##### sort in ascending of size ####
considered_community = sorted(considered_community, key=len)



############################################################################################################  
###      MIA Model with Tag    - community wise budget division     ######
############################################################################################################
    
print("*****************************************************************************************************")
print("***********   BaseLine TBIM-with community -- tag count specific prob setting  **********************")
print("*****************************************************************************************************") 
EI_budget = np.zeros(len(budget))
for b in range(len(budget)):
    start_ts = time.time()

    B_k = np.zeros(len(considered_community))
    remaining_budget = budget[b]
    print("*********************************************")
    print("Total Budget:", remaining_budget)
    for i in range(len(B_k)):
        allocated_budget = np.floor(budget[b] * len(considered_community[i]) / DG_prime.number_of_nodes())
        if allocated_budget <=  remaining_budget:
            B_k[i] = allocated_budget
            remaining_budget = remaining_budget - allocated_budget
        if i == len(B_k)-1:
            B_k[i] = B_k[i] + remaining_budget
    print("Community Budget: ", B_k)
    
    ### seed set and intial tag selection ##############
    SK_prime = []
    TK_prime = []
    for k in range(len(considered_community)):
        Bk_u = B_k[k]/2.0
        Bk_t = B_k[k]/2.0
    
        node_outdeg_list = sorted(list(DG_prime.out_degree(considered_community[k])) , key=lambda x: x[1], reverse=True)
        #print(node_outdeg_list)
        sorted_nodes= [node_outdeg_list[node][0] for node in range(len(node_outdeg_list))]
        #print(sorted_nodes)
    

        total_user_cost = 0.0
        for user in sorted_nodes:
            if total_user_cost + cost_per_user[user] <= Bk_u:
                total_user_cost = total_user_cost + cost_per_user[user]
                SK_prime.append(user)
        ### left out budget distribution
        if Bk_u - total_user_cost > 0:
            B_k[len(considered_community)-1] += (Bk_u - total_user_cost)
       
        
        community_1000tags = community_tag_count[np.abs(k - len(considered_community) + 1)].most_common(500)    ############################################### change
        #print(community_1000tags)
        sorted_1000tags =[]
        for tc in range(len(community_1000tags)):
            try:
                sorted_1000tags.append(top_1000_tags_map[ community_1000tags[tc][0]])
            except:
                continue
            

        total_tag_cost = 0.0
        for tag in sorted_1000tags:
            if tag not in TK_prime and total_tag_cost + cost_per_mtag[tag] <= Bk_t:
                total_tag_cost = total_tag_cost + cost_per_mtag[tag]
                TK_prime.append(tag)
        ### left out budget distribution
        if Bk_t - total_tag_cost > 0 :
            B_k[len(considered_community)-1] += (Bk_t - total_tag_cost)     
       
    ###### Edge prob calculation tag count specific ####################         
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
            
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]]['prob_weight_vec'][TK_prime]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )  
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    print("Number of edges: ", DG_prime_instance.number_of_edges() )
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, theta= 0.1, weight = 'tag_accum_prob_weight')
    
    EI_budget[b] = Expected_Influence(DG_prime_instance,  SK_prime , \
                   All_MIIA_DG, l=2, weight = 'tag_accum_prob_weight')
    end_ts = time.time()
    print("Expected Influence: ", EI_budget[b])
    result.append([EI_budget[b], len(SK_prime), end_ts-start_ts])
    #EI_setting_budget['count_prob']['comm'].append(EI_budget[b])
    
EI_setting_budget['count_prob'] = result
result = []


print("*****************************************************************************************************")
print("***********       BaseLine TBIM-with community -- trivalency prob setting      **********************")
print("*****************************************************************************************************") 
EI_budget = np.zeros(len(budget))
for b in range(len(budget)):
    start_ts = time.time()

    B_k = np.zeros(len(considered_community))
    remaining_budget = budget[b]
    print("*********************************************")
    print("Total Budget:", remaining_budget)
    for i in range(len(B_k)):
        allocated_budget = np.floor(budget[b] * len(considered_community[i]) / DG_prime.number_of_nodes())
        if allocated_budget <=  remaining_budget:
            B_k[i] = allocated_budget
            remaining_budget = remaining_budget - allocated_budget
    print("Community Budget: ", B_k)
    
    ### seed set and intial tag selection ##############
    SK_prime = []
    TK_prime = []
    for k in range(len(considered_community)):
        Bk_u = B_k[k]/2.0
        Bk_t = B_k[k]/2.0
    
        node_outdeg_list = sorted(list(DG_prime.out_degree(considered_community[k])) , key=lambda x: x[1], reverse=True)
        #print(node_outdeg_list)
        sorted_nodes= [node_outdeg_list[node][0] for node in range(len(node_outdeg_list))]
        #print(sorted_nodes)
    

        total_user_cost = 0.0
        for user in sorted_nodes:
            if total_user_cost + cost_per_user[user] <= Bk_u:
                total_user_cost = total_user_cost + cost_per_user[user]
                SK_prime.append(user)
        ### left out budget distribution
        if Bk_u - total_user_cost > 0:
            B_k[len(considered_community)-1] += (Bk_u - total_user_cost)
       
        
        community_1000tags = community_tag_count[np.abs(k - len(considered_community) + 1)].most_common(500)  ############################################### change
        #print(community_1000tags)
        sorted_1000tags =[]
        for tc in range(len(community_1000tags)):
            try:
                sorted_1000tags.append(top_1000_tags_map[ community_1000tags[tc][0]])
            except:
                continue
            

        total_tag_cost = 0.0
        for tag in sorted_1000tags:
            if tag not in TK_prime and total_tag_cost + cost_per_mtag[tag] <= Bk_t:
                total_tag_cost = total_tag_cost + cost_per_mtag[tag]
                TK_prime.append(tag)
        ### left out budget distribution
        if Bk_t - total_tag_cost > 0 :
            B_k[len(considered_community)-1] += (Bk_t - total_tag_cost)    
       
    ###### Edge prob calculation tag count specific ####################         
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
            
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]]['prob_weight_vec_tri'][TK_prime]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )  
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    print("Number of edges: ", DG_prime_instance.number_of_edges() )
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, theta= 0.1, weight = 'tag_accum_prob_weight')
    
    EI_budget[b] = Expected_Influence(DG_prime_instance,  SK_prime , \
                   All_MIIA_DG, l=2, weight = 'tag_accum_prob_weight')
    end_ts = time.time()
    print("Expected Influence: ", EI_budget[b])
    result.append([EI_budget[b], len(SK_prime), end_ts-start_ts])
    #EI_setting_budget['trivalency']['comm'].append(EI_budget[b])
    
    
EI_setting_budget['trivalency'] = result
result = []


print("*****************************************************************************************************")
print("***********    BaseLine TBIM-with community -- weighted cascade prob setting   **********************")
print("*****************************************************************************************************") 
EI_budget = np.zeros(len(budget))
for b in range(len(budget)):
    start_ts = time.time()

    B_k = np.zeros(len(considered_community))
    remaining_budget = budget[b]
    print("*********************************************")
    print("Total Budget:", remaining_budget)
    for i in range(len(B_k)):
        allocated_budget = np.floor(budget[b] * len(considered_community[i]) / DG_prime.number_of_nodes())
        if allocated_budget <=  remaining_budget:
            B_k[i] = allocated_budget
            remaining_budget = remaining_budget - allocated_budget
    print("Community Budget: ", B_k)
    
    ### seed set and intial tag selection ##############
    SK_prime = []
    TK_prime = []
    for k in range(len(considered_community)):
        Bk_u = B_k[k]/2.0
        Bk_t = B_k[k]/2.0
    
        node_outdeg_list = sorted(list(DG_prime.out_degree(considered_community[k])) , key=lambda x: x[1], reverse=True)
        #print(node_outdeg_list)
        sorted_nodes= [node_outdeg_list[node][0] for node in range(len(node_outdeg_list))]
        #print(sorted_nodes)
    

        total_user_cost = 0.0
        for user in sorted_nodes:
            if total_user_cost + cost_per_user[user] <= Bk_u:
                total_user_cost = total_user_cost + cost_per_user[user]
                SK_prime.append(user)
        ### left out budget distribution
        if Bk_u - total_user_cost > 0:
            B_k[len(considered_community)-1] += (Bk_u - total_user_cost)
       
        
        community_1000tags = community_tag_count[np.abs(k - len(considered_community) + 1)].most_common(500)   ###################################### change
        #print(community_1000tags)
        sorted_1000tags =[]
        for tc in range(len(community_1000tags)):
            try:
                sorted_1000tags.append(top_1000_tags_map[ community_1000tags[tc][0]])
            except:
                continue
            

        total_tag_cost = 0.0
        for tag in sorted_1000tags:
            if tag not in TK_prime and total_tag_cost + cost_per_mtag[tag] <= Bk_t:
                total_tag_cost = total_tag_cost + cost_per_mtag[tag]
                TK_prime.append(tag)
        ### left out budget distribution
        if Bk_t - total_tag_cost > 0 :
            B_k[len(considered_community)-1] += (Bk_t - total_tag_cost)  
       
    ###### Edge prob calculation tag count specific ####################         
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
            
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]]['prob_weight_vec_WC'][TK_prime]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )  
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    print("Number of edges: ", DG_prime_instance.number_of_edges() )
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, theta= 0.1, weight = 'tag_accum_prob_weight')
    
    EI_budget[b] = Expected_Influence(DG_prime_instance,  SK_prime , \
                   All_MIIA_DG, l=2, weight = 'tag_accum_prob_weight')
    end_ts = time.time()
    print("Expected Influence: ", EI_budget[b])
    result.append([EI_budget[b], len(SK_prime), end_ts-start_ts])
    #EI_setting_budget['WeightedCascade']['comm'].append(EI_budget[b])

   
EI_setting_budget['WeightedCascade'] = result
result = []



with open('result_3_HN_HT_comm.pickle', 'wb') as handle:
    pickle.dump(EI_setting_budget, handle)
