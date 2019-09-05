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

if len(sys.argv) == 3:
    budget = np.array([int(sys.argv[2])])
else:
    budget = np.linspace(1000.0, 8000.0, 8)

EI_setting_budget = dict()
#EI_setting_budget['count_prob']
#EI_setting_budget['trivalency']
#EI_setting_budget['WeightedCascade']


result = []

###### considered_community #######
##### sort in ascending of size ####   #################################################################################### change
considered_community = sorted(considered_community, key=len)   


def get_All_MIIA_DG(DG_prime, weight, TK_prime):
    try:
        del DG_prime_instance 
        DG_prime_instance = DG_prime.copy()
    except:
        DG_prime_instance = DG_prime.copy()
            
    for edge in DG_prime_instance.edges():
        p_vec = DG_prime_instance[edge[0]][edge[1]][weight][TK_prime]  
        p_vec = np.ones(len(p_vec)) - p_vec
        DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight']  = (1 - np.prod(p_vec) )  
    for edge in DG_prime.edges():
        if DG_prime_instance[edge[0]][edge[1]]['tag_accum_prob_weight'] <= 0:
            DG_prime_instance.remove_edge(edge[0], edge[1])
                
    All_MIIA_DG = {}
    for t in DG_prime.nodes():
        #print(t)
        All_MIIA_DG[t] = Maximum_Influence_In_Arborescence(DG_prime_instance, t, theta= 0.1, \
                                                           weight = 'tag_accum_prob_weight')
    return All_MIIA_DG, DG_prime_instance


def seed_tag_set_selection( DG_prime, TK_prime, Bk_u, Bk_t, sorted_1000tags, Initial_Tag, considered_community_nodes,\
                           cost_per_user, cost_per_mtag, weight = prob_weight_vec ):
    total_tag_cost = 0.0
    total_user_cost = 0.0
    S_k = []
    T_k = Initial_Tag.copy()
    Vk_G = considered_community_nodes.copy()
    
    no_user_to_Add = False
    no_tag_to_Add = False
    
    while Bk_u - total_user_cost> 0 and Bk_t - total_tag_cost > 0:
        
        All_MIIA_DG, DG_prime_instance = get_All_MIIA_DG(DG_prime, weight, T_k + TK_prime) ### change 07-05-2019 added + TK_prime
        sigma_STk = Expected_Influence(DG_prime_instance, S_k , All_MIIA_DG, l=2, weight='tag_accum_prob_weight')
        
        max_val = 0.0
        max_u = 0
        max_t = 0
        
        sorted_1000tags = [tg for tg in sorted_1000tags if total_tag_cost + cost_per_mtag[tg] < Bk_t and tg not in T_k + TK_prime]
        Vk_G = [ux for ux in Vk_G if total_user_cost + cost_per_user[ux] < Bk_u] 
        
        for tag in sorted_1000tags:
            if tag in T_k:
                continue
            
            new_T_k = T_k.copy()
            new_T_k.append(tag)
            try:
                del All_MIIA_DG, DG_prime_instance
                All_MIIA_DG, DG_prime_instance = get_All_MIIA_DG(DG_prime, weight, new_T_k + TK_prime)    ### change 07-05-2019 added + TK_prime
            except:
                All_MIIA_DG, DG_prime_instance = get_All_MIIA_DG(DG_prime, weight, new_T_k + TK_prime)     ### change 07-05-2019 added + TK_prime
            
            
            for user in Vk_G:
                if user in S_k:
                    continue
                new_S = S_k.copy()
                new_S.append(user)
                sigma_Sk_u = Expected_Influence(DG_prime_instance, new_S , All_MIIA_DG, \
                                                l=2, weight='tag_accum_prob_weight')
                new_val = (sigma_Sk_u - sigma_STk) / (cost_per_user[user] + cost_per_mtag[tag])
                #print("Old EI",sigma_Sk, "vertex", u, "new EI", sigma_Sk_u, "cost:", cost_per_user[u],\
                #     "new val:", new_val, "max_val:", max_val, "current Max u:", max_u )
                
                if new_val > max_val :
                    max_val = new_val
                    max_u = user
                    max_t = tag
        
        ###### update seed and tag set  ##############
        if max_u > 0 and max_t > 0:            
            if total_user_cost + cost_per_user[max_u] <= Bk_u:
                total_user_cost = total_user_cost + cost_per_user[max_u] 
                S_k.append(max_u)
                Vk_G.remove(max_u)
                #print("vertex added:", max_u)
            else:
                #print("Not able to add maximum:", max_u)
                Vk_G.remove(max_u)
                
            if total_tag_cost + cost_per_mtag[max_t] <= Bk_t:
                total_tag_cost = total_tag_cost + cost_per_mtag[max_t]
                T_k.append(max_t)
                sorted_1000tags.remove(max_t)
                #print("vertex added:", max_u)
            else:
                #print("Not able to add maximum:", max_u)
                sorted_1000tags.remove(max_t)
        
        ################  check for no user to add   ###################
        if len(Vk_G) > 0:
            if total_user_cost + min([ cost_per_user[_vx] for _vx in Vk_G]) > Bk_u:
                #print("Not able to add any vertex")
                no_user_to_Add = True
        else:
            no_user_to_Add = True

            
        ############### Check for no tag to add ##################
        if len(sorted_1000tags) > 0:
            if total_tag_cost + min([ cost_per_mtag[_vx] for _vx in sorted_1000tags]) > Bk_t:
                #print("Not able to add any vertex")
                no_tag_to_Add = True
        else:
            no_tag_to_Add = True
            
            
        ####### utilize left out tag budget  ####
        if (not no_tag_to_Add) and (no_user_to_Add):
            for tag in sorted_1000tags:
                if cost_per_mtag[tag] +  total_tag_cost < Bk_t and tag not in T_k:
                    total_tag_cost = total_tag_cost + cost_per_mtag[tag]
                    T_k.append(tag)
            no_tag_to_Add = True
            
        ###### utilize left out user budget  #####
        if (not no_user_to_Add) and (no_tag_to_Add):
            node_outdeg_list = sorted(list(DG_prime.out_degree(Vk_G)) , key=lambda x: x[1], reverse=True)
            Vk_G= [node_outdeg_list[node][0] for node in range(len(node_outdeg_list))]
            
            for user in Vk_G:
                if cost_per_user[user] +  total_user_cost < Bk_u and user not in S_k:
                    total_user_cost = total_user_cost + cost_per_user[user]
                    S_k.append(user)
            no_user_to_Add = True
            
        if no_user_to_Add and no_tag_to_Add:
            break
    
    
    
    rtb_S  =  Bk_u - total_user_cost
    rtb_T =  Bk_t - total_tag_cost
    
    return S_k, T_k, rtb_S, rtb_T
            

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
        allocated_budget = np.floor( budget[b] * len(considered_community[i]) / DG_prime.number_of_nodes() )
        if allocated_budget <=  remaining_budget:
            B_k[i] = allocated_budget
            remaining_budget = remaining_budget - allocated_budget
        if i == len(B_k)-1:    #################################################################################### change
            B_k[i] = B_k[i] + remaining_budget
    print("Community Budget: ", B_k)
    
    Bk_u = np.zeros(len(considered_community))
    Bk_t = np.zeros(len(considered_community))
    for k in range(len(considered_community)):
        Bk_u[k] = B_k[k]/2.0
        #Bk_u = B_k[k]/2.0
        Bk_t[k] = B_k[k]/2.0    
    
    ### seed set and intial tag selection ##############
    SK_prime = []
    TK_prime = []
    for k in range(len(considered_community)):
        Initial_Tag = []
        community_1000tags = community_tag_count[np.abs(k - len(considered_community) + 1)].most_common(500)  #################################################################################### change
        sorted_1000tags =[]
        for tc in range(len(community_1000tags)):
            try:
                sorted_1000tags.append(top_1000_tags_map[ community_1000tags[tc][0]])
            except:
                continue

        total_tag_cost = 0.0
        for tag in sorted_1000tags:
            if cost_per_mtag[tag] <= Bk_t[k] and tag not in TK_prime:   ### change 07-05-2019
                total_tag_cost = total_tag_cost + cost_per_mtag[tag]
                Bk_t[k] = Bk_t[k] - cost_per_mtag[tag]
                Initial_Tag.append(tag)  #### initial tag selection
                break
        if total_tag_cost == 0  and len(Initial_Tag) == 0:  
            continue
            
        S_com, T_com, rtb_S, rtb_T = seed_tag_set_selection( DG_prime, TK_prime, Bk_u[k], Bk_t[k], \
                                                sorted_1000tags, Initial_Tag,\
                                                considered_community[k], cost_per_user, cost_per_mtag, \
                                                weight = prob_weight_vec )
        
         
        ## update final seed set and tag set
        for sds in S_com:
            SK_prime.append(sds)
        for tgs in T_com:
            TK_prime.append(tgs)
        ## add remaining budget
        Bk_u[len(considered_community)-1] += rtb_S
        Bk_t[len(considered_community)-1] += rtb_T
        
       
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
        
    
    EI_budget[b] = Expected_Influence(DG_prime_instance,  SK_prime , \
                   All_MIIA_DG, l=2, weight = 'tag_accum_prob_weight')

    end_ts = time.time()
    
    print("seed set:", SK_prime, "length:", len(SK_prime))
    print("tag set:", TK_prime, "length:", len(TK_prime))
    print("Expected Influence: ", EI_budget[b])
    print("RESULT: ", EI_budget[b], len(SK_prime), end_ts - start_ts)
    result.append([EI_budget[b], len(SK_prime), end_ts - start_ts])
    #EI_setting_budget['count_prob']['comm-u1'].append(EI_budget[b])
    

print(result)


EI_setting_budget[ei_str] = result
result = []

if len(sys.argv) == 3:
    file_save = 'result_7_Max_EIT_comm_'+wt_flag.upper()+'_'+str(sys.argv[2])+'.pickle'
else:
    file_save = 'result_7_Max_EIT_comm_'+wt_flag.upper()+'.pickle'

with open(file_save, 'wb') as handle:
    pickle.dump(EI_setting_budget, handle)

