import community
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import random
import pickle

path='/home/suman/Desktop/BIM_with_Tag/Dataset/hetrec2011-delicious-2k/'
file = 'user_taggedbookmarks-timestamps.dat'
social_file = 'user_contacts.dat'
socialdata_df = pd.read_table(path+social_file, sep='\t')

G=nx.Graph()
for i in range(len(socialdata_df)):
    G.add_edge(socialdata_df.iloc[i]['userID'], socialdata_df.iloc[i]['contactID'])

print("Number of Nodes: ", G.number_of_nodes(), "Number of Edges: ", G.number_of_edges())

### Community Detection ######
partition = community.best_partition(G)

size = float(len(set(partition.values())))

community_nodes = []
count = 0
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    community_nodes.append(list_nodes)

community_nodes = sorted(community_nodes, key=len, reverse = True)

considered_community = []
for i in range(len(community_nodes)):
    if len(community_nodes[i]) >= 20:
        considered_community.append(community_nodes[i])


considered_nodes = set()
for i in range(len(considered_community)):
    for node in considered_community[i]:
        considered_nodes.add(node)

removed_nodes = set(G.nodes)-set(considered_nodes)

G_prime = G.copy()

G_prime.remove_nodes_from(list(removed_nodes))

print("considered Number of Nodes: ", G_prime.number_of_nodes(), "considered Number of Edges: ", G_prime.number_of_edges())
print("Total community considered: ", len(considered_community))

### Tag data reading
data_df = pd.read_table(path+file, sep='\t')

user_tag = defaultdict(set)
for i in range(len(data_df)):
    user_tag[data_df.iloc[i]['userID']].add(data_df.iloc[i]['tagID'])

user_tag_count = defaultdict(list)
for i in range(len(data_df)):
    user_tag_count[data_df.iloc[i]['userID']].append(data_df.iloc[i]['tagID'])

user_tag_count_1 = user_tag_count.copy()
for user in user_tag_count.keys():
    user_tag_count[user]=Counter(user_tag_count[user])

### tag count per community
community_tag_count = defaultdict(list)

for com in range(len(considered_community)):
    for node in community_nodes[com]:
        for tag in user_tag_count_1[node]:
            community_tag_count[com].append(tag)
            
for com in community_tag_count.keys():
    community_tag_count[com]=Counter(community_tag_count[com])


### Frequent tag count plot per community for top-5 community
freq_tags = set()
for i in range(5):
    most_freq = community_tag_count[i].most_common(5)
    for j in [str(a[0]) for a in most_freq]:
        freq_tags.add(j)

freq_tag_count_com = defaultdict(Counter)
for i in range(5):
    most_freq = community_tag_count[i].most_common(5)
    for j in range(len(most_freq)):
        freq_tag_count_com[ most_freq[j][0] ][i] = most_freq[j][1]

freq_tag_count_com_1 = defaultdict(Counter)
for i in range(5):
    for tag in freq_tags:
        freq_tag_count_com_1[int(tag)][i] = community_tag_count[i][int(tag)]

### plot
freq_tags = list(freq_tags)
community = ['Community-1', 'Community-2', 'Community-3', 'Community-4', 'Community-5']
pos = np.arange(len(freq_tags))
bar_width = 0.15

plt.figure(figsize=(20,10)) 
plt.bar(pos,[freq_tag_count_com_1[int(t)][0]/len(considered_community[0])  for t in freq_tags],bar_width) #, color='darkmagenta')
plt.bar(pos+1*bar_width,[freq_tag_count_com_1[int(t)][1]/len(considered_community[1])  for t in freq_tags],bar_width) #,color='plum')
plt.bar(pos+2*bar_width,[freq_tag_count_com_1[int(t)][2]/len(considered_community[2])   for t in freq_tags],bar_width)#,color='deeppink')
plt.bar(pos+3*bar_width,[freq_tag_count_com_1[int(t)][3]/len(considered_community[3])  for t in freq_tags],bar_width)#,color='pink') 
plt.bar(pos+4*bar_width,[freq_tag_count_com_1[int(t)][4]/len(considered_community[4])  for t in freq_tags],bar_width)#,color='lightsalmon')                
plt.xticks(pos+2*bar_width, freq_tags, fontsize = 25)
plt.yticks( fontsize = 25)
plt.xlabel('Frequent Tags', fontsize=40)
plt.ylabel('Normalized Tag Frequency', fontsize=40)
plt.title('Top-5 Frequent Tag Count per Community',fontsize=40)
plt.legend(community,loc=2, fontsize = 25)
plt.savefig("delicious_top_tag_comm.jpg")

###### Considered tag from the community

considered_tag_count = defaultdict(int)
for i in range(len(community_tag_count)):
    for tag in community_tag_count[i]:
        considered_tag_count[tag] = considered_tag_count[tag] + community_tag_count[i][tag]

considered_tag_count = Counter(considered_tag_count)


######## Considered only top 1000 tags
considered_tag_count_1 = considered_tag_count.most_common(1000)
top_1000_tags = [ considered_tag_count_1[i][0] for i in range(len(considered_tag_count_1))]
top_1000_tags.sort()

top_1000_tags_map = {}
for idx,tag in enumerate(top_1000_tags):
    top_1000_tags_map[tag] = idx


user_list = list(G_prime.nodes())
user_list.sort()

user_map = {}
for idx,user in enumerate(user_list):
    user_map[user] = idx

#### user tag matris creation 
user_tag_matrix = np.zeros((G_prime.number_of_nodes(), len(top_1000_tags)))
for user in user_list:
    for tag in list(user_tag_count[user].keys()):
        if tag in top_1000_tags:
            user_tag_matrix[user_map[user]][top_1000_tags_map[tag]] = user_tag_count[user][tag]

def set_probability_weight_vector(user_from_vec, user_to_vec):
    prob_weight_vec = user_from_vec - user_to_vec
    prob_weight_vec[prob_weight_vec <0 ] = 0
    user_from_vec = user_from_vec + np.ones(len(user_from_vec))
    prob_weight_vec = prob_weight_vec / user_from_vec
    
    return prob_weight_vec

del G
del considered_tag_count
## directed graph
DG_prime = G_prime.to_directed()

### probability weight vector setting three types:::::::
for edge in DG_prime.edges():
    #print(edge[0],edge[1])
    prob_weight_vec = set_probability_weight_vector( user_tag_matrix[user_map[edge[0]],:],  user_tag_matrix[user_map[edge[1]],:] )
    DG_prime[edge[0]][edge[1]]['prob_weight_vec'] = prob_weight_vec


for edge in DG_prime.edges():
    DG_prime[edge[0]][edge[1]]['prob_weight_vec_tri'] = np.random.choice([0, 0.1, 0.01, 0.001], \
                          size= len(DG_prime[edge[0]][edge[1]]['prob_weight_vec'] )  \
                         , replace = True, p=[0.4, 0.2, 0.2, 0.2])
 

for node in DG_prime.nodes():
    current = user_tag_matrix[user_map[node],:].copy()
    current[current > 0] = 1
    all_indeg_tag = user_tag_matrix[[user_map[i] for i in list(DG_prime.predecessors(node))], :]
    all_indeg_tag[all_indeg_tag > 0] = 1
    sum_tag = np.sum(all_indeg_tag, axis = 0)
    
    prob_vec = current/sum_tag
    prob_vec[prob_vec == np.inf] = 0.0
    prob_vec[np.isnan(prob_vec)] = 0.0
    
    for inc_node in list(DG_prime.predecessors(node)):
        DG_prime[inc_node][node]['prob_weight_vec_WC'] = prob_vec



############################################################################################################
## random cost selection from [50 , 100] for each user
cost_per_user = {}
for s in DG_prime.nodes():
    cost_per_user[s] = np.random.randint(50, 100)
 
 ############################################################################################################
## random cost selection from [25, 50] for each tag
cost_per_mtag = {}
for tag in top_1000_tags_map.keys():
    cost_per_mtag[top_1000_tags_map[tag]] = np.random.randint(25, 50)




nx.write_gpickle(DG_prime, "DG_prime.gpickle")
np.save("user_tag_matrix.npy", user_tag_matrix)
with open('considered_community.pickle', 'wb') as handle:
    pickle.dump(considered_community, handle)

with open('user_map.pickle', 'wb') as handle:
    pickle.dump(user_map, handle)

with open('top_1000_tags_map.pickle', 'wb') as handle:
    pickle.dump(top_1000_tags_map, handle)

with open('cost_per_user.pickle', 'wb') as handle:
    pickle.dump(cost_per_user, handle)

with open('cost_per_mtag.pickle', 'wb') as handle:
    pickle.dump(cost_per_mtag, handle)

with open('community_tag_count.pickle', 'wb') as handle:
    pickle.dump(community_tag_count, handle)
with open('considered_tag_count_1.pickle', 'wb') as handle:
    pickle.dump(considered_tag_count_1, handle)






