import community
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import random
import pickle
import time
import sys

from MIA_calculate import MIP_path_to_target
from MIA_calculate import Maximum_Influence_In_Arborescence
from MIA_calculate import activation_probability
from MIA_calculate import Expected_Influence

######################### Load  #################################

wt_flag = str(sys.argv[1])
if wt_flag.upper() == 'CP':
    prob_weight_vec = 'prob_weight_vec'
    ei_str = 'count_prob'
    print("computing Probability setting for $$$$$$$$$$ ", ei_str)
elif wt_flag.upper() == 'TRI':
    prob_weight_vec = 'prob_weight_vec_'+wt_flag.lower()
    ei_str = 'trivalency'
    print("computing Probability setting for $$$$$$$$$$ ", ei_str)
elif wt_flag.upper() == 'WC':
    prob_weight_vec = 'prob_weight_vec_'+wt_flag.upper()
    ei_str = 'WeightedCascade'
    print("computing Probability setting for $$$$$$$$$$ ", ei_str)
else:
    print('Give correct weight from CP, TRI, WC')


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
##### sort in ascending of size ####   #################################################################################### change
considered_community = sorted(considered_community, key=len)   


def seed_select_per_community(DG, All_MIIA_DG, B_k, Vk_G, cost_per_user, weight):
    l=2
    S_k = []
    total_user_cost = 0.0
    while B_k - total_user_cost> 0:
        max_val = 0.0
        max_u = 0
        sigma_Sk = Expected_Influence(DG, S_k , All_MIIA_DG, l, weight)
        #print("**********************************************", sigma_Sk)
        Vk_G = [_vx for _vx in Vk_G if cost_per_user[_vx] <= B_k - total_user_cost]
        for u in Vk_G:
            if u in S_k:
                continue
            new_S = S_k.copy()
            new_S.append(u)
            sigma_Sk_u = Expected_Influence(DG, new_S , All_MIIA_DG, l, weight)
            new_val = (sigma_Sk_u - sigma_Sk) /cost_per_user[u]
            #print("Old EI",sigma_Sk, "vertex", u, "new EI", sigma_Sk_u, "cost:", cost_per_user[u],\
            #     "new val:", new_val, "max_val:", max_val, "current Max u:", max_u )
            
            if new_val > max_val :
                max_val = new_val
                max_u = u
                
        if max_u > 0:            
            if total_user_cost + cost_per_user[max_u] <= B_k:
                total_user_cost = total_user_cost + cost_per_user[max_u] 
                S_k.append(max_u)
                Vk_G.remove(max_u)
                #print("vertex added:", max_u)
            else:
                #print("Not able to add maximum:", max_u)
                Vk_G.remove(max_u)
        ##### break : if no vertex can be added
        #check_flag = [ cost_per_user[_vx] + total_user_cost - B_k for _vx in Vk_G]
        #flag = all(item > 0 for item in check_flag)
        print(S_k)
        if len(Vk_G) > 0:
            if total_user_cost + min([ cost_per_user[_vx] for _vx in Vk_G]) > B_k:
                #print("Not able to add any vertex")
                break
        else:
            break
        
        
    return S_k, (B_k - total_user_cost)




def seed_set_selection(DG, All_MIIA_DG, Bk_u, considered_community, cost_per_user, weight):
    l=2
    S = []
    left_out_budget = 0.0
    for k in range(len(considered_community)):
        print("community:", k)

        if k == len(considered_community) - 1:
            S_k, ret_budget = seed_select_per_community(DG, All_MIIA_DG, Bk_u[k] + left_out_budget,  considered_community[k], cost_per_user, weight)
        else:      
            S_k, ret_budget = seed_select_per_community(DG, All_MIIA_DG, Bk_u[k],  considered_community[k], cost_per_user, weight)
            left_out_budget = left_out_budget + ret_budget
        for cu in S_k:
            S.append(cu)
    return S      



print("*****************************************************************************************************")
print("***********   BaseLine TBIM-with community (Incremental Greedy)-- **********************")
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
        if i == len(B_k)-1:    #################################################################################### change
            B_k[i] = B_k[i] + remaining_budget
    print("Community Budget: ", B_k)
    
    ### seed set and intial tag selection ##############
    #SK_prime = []
    TK_prime = []
    Bk_u = np.zeros(len(considered_community))
    for k in range(len(considered_community)):
        Bk_u[k] = B_k[k]/2.0
        #Bk_u = B_k[k]/2.0
        Bk_t = B_k[k]/2.0      
        
        community_1000tags = community_tag_count[np.abs(k - len(considered_community) + 1)].most_common(500)  #################################################################################### change
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
        if Bk_t - total_tag_cost > 0 :  #################################################################################### change
            B_k[len(considered_community)-1] += (Bk_t - total_tag_cost)  
       
    ###### Edge prob calculation tag count specific ####################         
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
            
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]][prob_weight_vec][TK_prime]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )  
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
    print("Number of edges: ", DG_prime_instance.number_of_edges() )
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, theta= 0.1, \
                                                           weight = 'tag_accum_prob_weight')
        
    
    
    SK_prime = seed_set_selection(DG_prime_instance, All_MIIA_DG, Bk_u, considered_community, cost_per_user, \
                                  weight = 'tag_accum_prob_weight')
    print("seed set:", SK_prime, "length:", len(SK_prime))
    EI_budget[b] = Expected_Influence(DG_prime_instance,  SK_prime , \
                   All_MIIA_DG, l=2, weight = 'tag_accum_prob_weight')

    end_ts = time.time()

    print("Expected Influence: ", EI_budget[b])
    result.append([EI_budget[b], len(SK_prime), end_ts - start_ts])
    #EI_setting_budget['count_prob']['comm-u1'].append(EI_budget[b])
    
EI_setting_budget[ei_str] = result
result = []

with open('result_4_Max_EI_comm_'+wt_flag.upper()+'.pickle', 'wb') as handle:
    pickle.dump(EI_setting_budget, handle)

