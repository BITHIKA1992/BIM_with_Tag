############################################################################################################  
###      MIA Model with Tag    - baseline- equal budget distribution for tag and user selection      ######
############################################################################################################

def MIP_path_to_target(DG, t, weight = 'dummy_weight'):
    #path_list = []
    MIP_Path = {}
    for a in list(DG.predecessors(t)):
        path_from_a = [a, t]
        #path_list.append(path_from_a)
        prob_at = DG[a][t][weight]
        MIP_Path[(a,t)] = (path_from_a, prob_at)
        for b in list(DG.predecessors(a)):
            path_from_b = path_from_a.copy()
            prob_ba = prob_at * DG[b][a][weight]
            if b not in path_from_b:
                path_from_b.insert(0, b)
                try:
                    current_prob = MIP_Path[(b,t)][1] 
                    if current_prob < prob_ba:
                        MIP_Path[(b,t)] = (path_from_b, prob_ba)
                except:
                    MIP_Path[(b,t)] = (path_from_b, prob_ba)
                
                #path_list.append(path_from_b)
                for c in list(DG.predecessors(b)):
                    path_from_c= path_from_b.copy()  
                    prob_cb = prob_ba * DG[c][b][weight]
                    if c not in path_from_c:
                        path_from_c.insert(0, c) 
                        #path_list.append(path_from_c)
                        try:
                            current_prob_1  = MIP_Path[(c,t)][1] 
                            if current_prob_1 < prob_cb:
                                MIP_Path[(c,t)] = (path_from_c, prob_cb)
                        except:
                            MIP_Path[(c,t)] = (path_from_c, prob_cb)
                                
                    del path_from_c, prob_cb
            del path_from_b, prob_ba
        del path_from_a, prob_at
        
    return MIP_Path
    


def Maximum_Influence_In_Arborescence(DG, t, theta, weight = 'dummy_weight'):
    MIIA = {}
    MIP_target_Path = MIP_path_to_target(DG, t, weight)
    for st in MIP_target_Path.keys():
        MIP_Path, MIP_prob = MIP_target_Path[st]
        if MIP_prob >= theta:
            MIIA[st] = (MIP_Path, MIP_prob)
    return MIIA
            


def activation_probability(DG, SeedSet, t, All_MIIA, l, weight):
    if l == 0:
        return 0.0    
    
    if t in SeedSet:
        return 1.0
    elif len(list(DG.predecessors(t))) == 0:
        return 0.0
    else:
        product_term = 1.0
        for w in list(DG.predecessors(t)):
            new_l = l - 1
            try: 
                pp_wt = All_MIIA[t][(w,t)][1]
            except:
                pp_wt = 0.0
            product_term = product_term*(1.0-(activation_probability(DG, SeedSet, w, All_MIIA, new_l, weight)* pp_wt))
        ap_t = 1.0 - product_term
        return ap_t
            
    


def Expected_Influence(DG, SeedSet, All_MIIA, l=3, weight= 'dummy_weight'):
    EI = 0.0
    for s in DG.nodes():
        #print('node :', s)
        EI = EI + activation_probability(DG, SeedSet, s, All_MIIA, l, weight)
        
    return EI
