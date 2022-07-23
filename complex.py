# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:51:38 2022

@author: Claudia Ghinato
"""

# COMPLEX NETWORK PROJECT - NODE2VEC

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from itertools import product
import pandas as pd
import scipy
import time
import fastnode2vec
from fastnode2vec import Graph, Node2Vec
import os
#os.getcwd()
#os.listdir()
#os.chdir("path_to_the_fold_of_interest")
os.chdir("...path...")

import random
seed=18
random.seed(seed)

import networkx as nx
from gensim.models import Word2Vec
import stellargraph as sg
from stellargraph.data import BiasedRandomWalk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#%%
# VISUALIZZAZIONE DATI Hi-C 


hic_dati = np.loadtxt('cancer_hic.txt', dtype= int, delimiter= ',')


dim=len(hic_dati)

plt.imshow(np.log10(hic_dati +1))
plt.colorbar()
plt.show()

#%%
# Hi-C OUTLIERS

chr_zone=[(0,249),(250,421),(422,577),(578,713),(714,777)]
boundary_end= [a_tuple[1] for a_tuple in chr_zone] #list of second terms in the chromosome boundary indexes
boundary_start= [a_tuple[0] for a_tuple in chr_zone]
len_chrom=[a_tuple[1]-a_tuple[0] for a_tuple in chr_zone]

def out_find(net, chrom:list): # net=original network, chrom=list of tuples for boundary indexes for each chromosome
    out_net=np.zeros((len(net),len(net)), dtype=(int))
    len_chrom=[a_tuple[1]-a_tuple[0] for a_tuple in chrom] # list of chromosomes' length
    boundary_end= [a_tuple[1] for a_tuple in chrom] #list of seconds terms in the chromosome boundary indexes
    boundary_start= [a_tuple[0] for a_tuple in chrom]
    start_point=0
    for n in range(0,len(len_chrom)): # iteration over the number of chromosomes
        for j in range(start_point, start_point+len_chrom[n]): #column
            for i in range(boundary_end[n], len(net)): #row
                out_net[i,j]=net[i,j]
                out_net[j,i]=net[i,j]
        start_point+=len_chrom[n]
    threshold_val=np.quantile(out_net,0.90)
    out_net= np.where(out_net > threshold_val, out_net, 0)
    return(out_net)
    
hic_true_out_net= out_find(hic_dati,chr_zone)

# removal of residual noise

for j in range(310, boundary_end[1]):
    for i in range(boundary_start[2], boundary_end[2]):
        hic_true_out_net[i,j]=0
        hic_true_out_net[j,i]=0
for j in range(boundary_start[1], boundary_end[1]):
    for i in range(480,boundary_end[2]):
        hic_true_out_net[i,j]=0
        hic_true_out_net[j,i]=0

plt.imshow(np.log10(hic_true_out_net +1))
plt.colorbar()
plt.show()

#%%
# Hi-C outliers in list of list (str) format for using it in node2vec and making comparisons
# Analysis on chr6 and X (2nd and 3rd one)

zona_6X = np.arange(boundary_start[1],boundary_end[2], dtype=int)

net_6X = hic_dati[np.ix_(zona_6X,zona_6X)]

ticks=np.arange(0, len(zona_6X), 50)
t_lab=np.arange(boundary_start[1], len(zona_6X)+boundary_start[1], 50)

plt.imshow(np.log10(net_6X +1))
plt.colorbar()
plt.xticks(ticks, t_lab)
plt.yticks(ticks, t_lab)
plt.show()

net_6X_out = hic_true_out_net[np.ix_(zona_6X,zona_6X)]
plt.imshow(np.log10(net_6X_out +1))
plt.colorbar()
plt.xticks(ticks, t_lab)
plt.yticks(ticks, t_lab)
plt.show()


hic_out_list_of_lists=[]

for j in range(boundary_start[1],boundary_end[1]):
    a=[]
    for i in range(j,boundary_end[2]):
        if(hic_true_out_net[i,j]!=0):
            a.append(str(i))
    hic_out_list_of_lists.append(a)

# hic_out_list_of_lists contains empty lists, where there are no outliers --> these nodes should not be given in input to node2vec
# it is required a "cleaning" 

hic_out_list=[]

for n in range(boundary_start[1],boundary_end[1]):
    if len(hic_out_list_of_lists[n-boundary_start[1]])!=0:
        hic_out_list.append(str(n))


log_out_val=[]
for j in range(boundary_start[1],boundary_end[1]):
    for i in range(j,boundary_end[2]):
        if(hic_true_out_net[i,j]!=0):
            log_out_val.append(np.log10(hic_true_out_net[i,j]+1))

mean_log_out_val=np.mean(log_out_val)
min_log_out_val=min(log_out_val)


#%%        
# ER TOY NETWORK

#len_chrom=[a_tuple[1]-a_tuple[0] for a_tuple in chr_zone]
dim_toy= len_chrom[1]+len_chrom[2]
toy_net= np.zeros((dim_toy,dim_toy), dtype=int)


# function for obtaining an upper triangular matrix (0s on the main diagonal)
# the function overwrite the input matrix
def triupper(prob, matrix):
    row= matrix.shape[0] # number of rows 
    col=matrix.shape[1]  # number of columns
    for i in range(0,row):
        for j in range(0,col):
            if (i>j or i==j):
                matrix[i][j]=0
            else:
                if (random.uniform(0,1)> 1-prob): # probability of having a link is greater than not having it
                    matrix[i][j]=1
                else:
                    matrix[i][j]=0
                    
P_out=0.65  # inter-blocks link probability  
       
dim_out= 50 # dim outlier block (square matrix)
aux= np.zeros((dim_out,dim_out),dtype=(int)) # auxiliary matrix to generate outliers block
triupper(P_out,aux)  ## new auxiliary matrix
blocco_out= np.add(aux, np.transpose(aux)) ## symmetrization
# multiplying for the average weight of outlier links
sum_true_out=sum(sum(hic_true_out_net))
n_out=0
for i in range(0,dim):
    for j in range(0,dim):
        if hic_true_out_net[i,j]!= 0:
            n_out+=1
mean_freq_out= int(sum_true_out/n_out)
blocco_out= blocco_out* mean_freq_out


P_chr=0.8

aux_6= np.zeros((len_chrom[1],len_chrom[1]),dtype=(int))
triupper(P_chr,aux_6)
blocco_6= np.add(aux_6, np.transpose(aux_6))
sum_chr=sum(sum(net_6X[np.ix_(np.arange(0,250),np.arange(0,250))]))
blocco_6=blocco_6*(int(sum_chr/62500))

aux_X=np.zeros((len_chrom[2],len_chrom[2]),dtype=(int))
triupper(P_chr,aux_X)
blocco_X= np.add(aux_X, np.transpose(aux_X))
blocco_X=blocco_X*(int(sum_chr/62500))

zona_1= np.arange(0, int(len_chrom[1]),dtype=int) #position indexes for 1st block in toy_net for chromosom6 6 (the 2nd one)
zona_2 = np.arange(int(len_chrom[1]), (int(len_chrom[1])+int(len_chrom[2])),dtype=int)
index1_out= np.arange(int(len_chrom[1]), (int(len_chrom[1])+int(dim_out)), dtype=int)
index2_out= np.arange(0,dim_out, dtype=int)

toy_net[np.ix_(zona_1, zona_1)]= blocco_6
toy_net[np.ix_(zona_2, zona_2)]= blocco_X
toy_net[np.ix_(index1_out,index2_out)]= blocco_out
toy_net[np.ix_(index2_out,index1_out)]= np.transpose(blocco_out)

plt.imshow(np.log10(toy_net +1))
plt.colorbar()
plt.show()

#%%
## Outliers toy network - data preparation

toy_out_list_of_lists=[]


for j in range(0,dim_out):
    a=[]  # a list for each node in the 1st chromosome having outlier links 
    for i in range(int(len_chrom[1]), int(len_chrom[1])+int(dim_out)):
        if (toy_net[i,j]!=0):
           a.append(str(i))
    
    toy_out_list_of_lists.append(a)

# toy_out_list is a list of lists. A list for each considered node in 1st chromosome having links outliers
# (i.e., 50 lists as the size of the outlier block)
# Each element of the list is a list containing the nodes of the other chromosome with which there is a link

toy_out_list=[]
for n in range(0,dim_out):
    if (len(toy_out_list_of_lists[n])!=0):
        toy_out_list.append(str(n))
   
        
#%%  
# Node2vec on ER network as parameters p, q, and walk length vary
# We obtain that all the combinations of parameters lead to the same results
# (Check len(good_rows_ER)==len(good_tries_ER)==len(params_ER))
'''
# Node2vec on ER toy network

# list of tuples of links for graph generation
toy_net_list_tuples=[]

for i in range(0,dim_toy):
    for j in range(i,dim_toy):
        if(toy_net[i,j]!=0):
            toy_net_list_tuples.append((str(i), str(j), toy_net[i,j]))
                    
            
graph_ER = Graph(toy_net_list_tuples, directed=False, weighted=True)       

start_time_ER = time.time()

good_tries_ER = np.array([['p', 'q', 'walk length', 'n_id_links_out', 'n_id_out_nodes', 'n_out', 'n_id_nodes_out_%']])

range_p_ER= np.array([0.01,0.25,0.5,0.75,1,1.25,1.5,2,5,10])
range_q_ER= np.array([0.01,0.25,0.5,0.75,1,1.25,1.5,2,5,10])
range_walk_len_ER= np.array([50,100,150,200,250,300,400,500],dtype=int)

params_ER= np.array(list(product(range_p_ER, range_q_ER, range_walk_len_ER)),dtype="float32")

n_out_ER= len(toy_out_list)

for n,pt in enumerate(params_ER):
    p=pt[0]
    q=pt[1]
    walk_len=pt[2]


    n2v_ER = Node2Vec(graph_ER, dim=10, walk_length=walk_len, context=300, p= p, q= q, workers=1)
    #n2v_ER.train(epochs=100)
    
    similar_output_node2_ER=[]  
    
    Similar_node2_ER=[]  
    
    for node in toy_out_list:
        similar_output_node2_ER.append( n2v_ER.wv.most_similar(str(node),topn=30))
      
        
    for lista in similar_output_node2_ER:          
        a=[]    
        for n in lista:     
            a.append(n[0])       
            
        Similar_node2_ER.append(a)
        
    #comparing results
    
    Identified_Outliers_node2_ER=[]
    
    n_id_links_out_ER=0  # number of identified link outliers
    
    for i in range(len(toy_out_list)):
        Identified_Outliers_node2_ER.append(list(set(Similar_node2_ER[i]).intersection(toy_out_list_of_lists[i])))
        n_id_links_out_ER+= len(list(set(Similar_node2_ER[i]).intersection(toy_out_list_of_lists[i])))
    
    nodes_out_ER=[]
    for lista in Identified_Outliers_node2_ER:
        for node in lista:
            nodes_out_ER.append(node)

    nodes_out_ER=list(set(nodes_out_ER))

    n_id_out_nodes_ER=len(nodes_out_ER)
    
    
    if(n_id_links_out_ER !=0):
        print('ER toy-network ok con p={}, q={}, walk_len= {}, n_links_out_id={}, n_id_out_nodes={}, out_nodes_id_%={}'.format(p,q,walk_len, n_id_links_out_ER, n_id_out_nodes_ER, n_id_out_nodes_ER/n_out_ER*100) )
        good_tries_ER= np.append(good_tries_ER, [[p, q, walk_len, n_id_links_out_ER, n_id_out_nodes_ER, n_out_ER, n_id_out_nodes_ER/n_out_ER*100]], axis=0)    
    
    


good_tries_ER = np.delete(good_tries_ER, (0), axis=0) 

good_tries_ER= pd.DataFrame(good_tries_ER,columns= [ 'p', 'q', 'walk length', 'n_id_links_out', 'n_id_out_nodes', 'n_out', 'n_id_nodes_out_%'] )

print("Max number of identified outlier links:{}".format(max(good_tries_ER['n_id_links_out'])))
print("Max number of identified outlier nodes:{}".format(max(good_tries_ER['n_id_out_nodes']))) 


good_rows_ER=good_tries_ER.index[good_tries_ER['n_id_out_nodes'] == max(good_tries_ER['n_id_out_nodes'])].tolist()        

print("--- %s seconds ---" % (time.time() - start_time_ER))

'''

#%%
# Node2vec on ER toy network

# list of tuples of links for graph generation
toy_net_list_tuples=[]

for i in range(0,dim_toy):
    for j in range(i,dim_toy):
        if(toy_net[i,j]!=0):
            toy_net_list_tuples.append((str(i), str(j), toy_net[i,j]))
            
            
graph_ER = Graph(toy_net_list_tuples, directed=False, weighted=True)       

start_time_ER = time.time()

good_tries_ER = np.array([['topn', 'n_id_out_nodes', 'n_out', 'n_id_nodes_out_%']])

p=0.25
q=1.25
walk_len=250
    
n2v_ER = Node2Vec(graph_ER, dim=10, walk_length=walk_len, context=10, p= p, q= q, workers=1)
#n2v_ER.train(epochs=100)
# context=300 dà gli stessi risultati
    

range_topn=np.array([5,10,15,20,25,30,40,50,60], dtype=int)

n_out_ER= len(toy_out_list)

for value in range_topn:
    topn=value
    
    similar_output_node2_ER=[]  
    # list as long as the number of nodes in 1st chromosome with outlier links
    # Each element in the list is a list of tuples: a tuple for each one of the identified most similar nodes

    
    Similar_node2_ER=[]  
    # List of lists. As many lists as the number of nodes in the 1st chr. having outlier links
    # For each of these nodes, there is a list of its most similar nodes
    
    for node in toy_out_list:
        similar_output_node2_ER.append( n2v_ER.wv.most_similar(str(node),topn=topn))
      
        
    for lista in similar_output_node2_ER:   # a list of tuples for each input node 
        a=[]    
        for n in lista:     # n=tuple in a list 
            a.append(n[0])        # considering just most similar nodes, negletting the 2nd values in output
            
        Similar_node2_ER.append(a)
        
    #comparing results
    
    Identified_Outliers_node2_ER=[]
    
    n_id_links_out_ER=0  # number of identified links outlier
    
    for i in range(len(toy_out_list)):
        Identified_Outliers_node2_ER.append(list(set(Similar_node2_ER[i]).intersection(toy_out_list_of_lists[i])))
        n_id_links_out_ER+= len(list(set(Similar_node2_ER[i]).intersection(toy_out_list_of_lists[i])))
    
    nodes_out_ER=[]
    for lista in Identified_Outliers_node2_ER:
        for node in lista:
            nodes_out_ER.append(node)

    nodes_out_ER=list(set(nodes_out_ER))

    n_id_out_nodes_ER=len(nodes_out_ER)
    
    
    if(n_id_links_out_ER !=0):
        print('ER toy-network ok con topn= {}, n_links_out_id={}, n_id_out_nodes={}, out_nodes_id_%={}'.format(topn,
                                             n_id_links_out_ER, n_id_out_nodes_ER, n_id_out_nodes_ER/n_out_ER*100) )
        good_tries_ER= np.append(good_tries_ER, [[int(topn), int(n_id_out_nodes_ER), 
                                                  int(n_out_ER), int(n_id_out_nodes_ER/n_out_ER*100)]], axis=0)    
    

good_tries_ER = np.delete(good_tries_ER, (0), axis=0) 

good_tries_ER= pd.DataFrame(good_tries_ER,columns= [ 'topn', 'n_id_out_nodes', 'n_out', 'n_id_nodes_out_%'] )

print("Max number of identified outlier nodes:{}".format(max(good_tries_ER['n_id_out_nodes']))) 

good_rows_ER=good_tries_ER.index[good_tries_ER['n_id_out_nodes'] == max(good_tries_ER['n_id_out_nodes'])].tolist()        

print("--- %s seconds ---" % (time.time() - start_time_ER))


#%%
#Plot n_id_out VS topn ER toy network

plt.plot(good_tries_ER['topn'],good_tries_ER['n_id_nodes_out_%'], 'ro')
plt.xlim=(0,max(range_topn))
plt.ylim=(0,100)
plt.xlabel('topn value')
plt.ylabel('Nodes outliers identified (%)')
plt.title('Toy-network')
plt.grid(True)
plt.show()


#%%
# k-means clustering on ER network
  
df_ER=[]
for j in range(0,dim_toy):
    for i in range(j,dim_toy):
        df_ER.append([str(j),str(i),toy_net[i,j]])

df_ER=pd.DataFrame(df_ER)
df_ER.columns=['source','target','weight']
    
graph_ER=sg.StellarGraph(edges=df_ER, is_directed=False)


# n = number of random walks per node
rw = BiasedRandomWalk(graph_ER, p = 0.25, q = 1.25, n = 30, length = 250, 
                      seed=42, weighted = True)

walks = rw.run(nodes=list(graph_ER.nodes())
               # root nodes
              )

# we pass the random walks to a list
str_walks = [[str(n) for n in walk] for walk in walks]
    
model_ER = Word2Vec(str_walks, vector_size=10, window=10, min_count=1, sg=1, workers=1, epochs=1)
#vector_size is the number of features the code look for
   
model_ER.wv.save_word2vec_format("EMBEDDING_ER")
# Save model for later use
model_ER.save("EMBEDDING_MODEL_ER")
# Retrieve node embeddings and corresponding subjects
node_ids = model_ER.wv.index_to_key  # list of node IDs
node_embeddings = (model_ER.wv.vectors) 

    
# k means
    
distortions = []
K = range(1,25)
for k in K:
    k_cluster = KMeans(n_clusters=k, max_iter=500, random_state=42).fit(node_embeddings)
    k_cluster.fit(node_embeddings)
    distortions.append(k_cluster.inertia_)
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for ER network')
plt.show()

#%%
# k-means dots graph for ER

best_k_ER=15
k_cluster = KMeans(n_clusters=best_k_ER, max_iter=500, random_state=42).fit(node_embeddings)
kmeans_labels = k_cluster.labels_
nodes_labels = pd.DataFrame(zip(node_ids, kmeans_labels),columns = ['node_ids','kmeans'])


# fit our embeddings with t-SNE
trans = TSNE(n_components = 2, early_exaggeration = 12,
                  perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                  learning_rate = 200.0, random_state = 412)
node_embeddings_2d = trans.fit_transform(node_embeddings)

# create the dataframe that has information about the nodes and their x and y coordinates
data_tsne = pd.DataFrame(zip(node_ids, list(node_embeddings_2d[:,0]),list(node_embeddings_2d[:,1])),
                        columns = ['node_ids','x','y'])
data_tsne = pd.merge(data_tsne, nodes_labels, left_on='node_ids', right_on='node_ids',
                how = 'left')

# plot using seaborn
import seaborn as sns
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne, x='x', y='y',hue='kmeans', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec clusters with k-means on ER')
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


## True network representation
ER_labels=np.zeros(len(node_ids))

i=0
for node in node_ids:
    if node in toy_out_list:
        ER_labels[i]=0 # node in the 1st chromosome with outlier links
    elif node in nodes_out_ER:
        ER_labels[i]=1 # node in the 2nd chromosome with outlier links
    elif (int(node)<= zona_1[-1] and node not in toy_out_list):
        ER_labels[i]=2 # node in 1st chromosome with no links outlier
    else:
        ER_labels[i]=3 # node in 2nd chromosome with no link outlier
    i+=1

true_ER_labels= pd.DataFrame(zip(node_ids, ER_labels),columns = ['node_ids','ER labels']) 

data_tsne = pd.DataFrame(zip(node_ids, list(node_embeddings_2d[:,0]),list(node_embeddings_2d[:,1])),
                        columns = ['node_ids','x','y'])
data_tsne = pd.merge(data_tsne, true_ER_labels, left_on='node_ids', right_on='node_ids',
                how = 'left')
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne, x='x', y='y',hue='ER labels', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec embedding of the ER network')
plt.legend(labels=['chr.2','out chr.1', 'out chr.2', 'chr.1'])
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()



#%%
# Toy network simplified

seed=79
random.seed(seed)

#CASE 1 - same size blocks

dim_toy2=400
toy_net2= np.zeros((dim_toy2,dim_toy2), dtype=int)

P_noise=0.3
P_blocks=0.85

dim_blocks=200
#block 1 case 1
aux_b11=np.zeros((dim_blocks,dim_blocks),dtype=(int)) 
triupper(P_blocks, aux_b11)
block11=np.add(aux_b11, np.transpose(aux_b11))

#block 2 case 1
aux_b21=np.zeros((dim_blocks,dim_blocks),dtype=(int)) 
triupper(P_blocks, aux_b21)
block21=np.add(aux_b21, np.transpose(aux_b21))

#noise directly in the network
aux_n1=np.zeros((dim_toy2,dim_toy2),dtype=(int)) 
triupper(P_noise, aux_n1)
toy_net2=np.add(aux_n1, np.transpose(aux_n1))

zona_b11= np.arange(0, dim_blocks,dtype=int)
zona_b21 = np.arange(dim_blocks, dim_toy2,dtype=int)

toy_net2[np.ix_(zona_b11, zona_b11)]= block11
toy_net2[np.ix_(zona_b21, zona_b21)]= block21

plt.imshow(toy_net2)
#plt.colorbar()
plt.show()


#CASE 2 - blocks of sizes very different

toy_net2bis= np.zeros((dim_toy2,dim_toy2), dtype=int)

dim_block1=80
dim_block2=320

# block 1 case 2
aux_b12=np.zeros((dim_block1,dim_block1),dtype=(int)) 
triupper(P_blocks, aux_b12)
block12=np.add(aux_b12, np.transpose(aux_b12))

#block 2 case 2
aux_b22=np.zeros((dim_block2,dim_block2),dtype=(int)) 
triupper(P_blocks, aux_b22)
block22=np.add(aux_b22, np.transpose(aux_b22))

#block noise directly in the network
aux_n2=np.zeros((dim_toy2,dim_toy2),dtype=(int)) 
triupper(P_noise, aux_n2)
toy_net2bis=np.add(aux_n2, np.transpose(aux_n2))

zona_b12= np.arange(0, dim_block1,dtype=int)
zona_b22 = np.arange(dim_block1, dim_toy2,dtype=int)

toy_net2bis[np.ix_(zona_b12, zona_b12)]= block12
toy_net2bis[np.ix_(zona_b22, zona_b22)]= block22


plt.imshow(toy_net2bis)
#plt.colorbar()
plt.show()


#%%
# Toy network analysis - CASE 1
# k-means clustering 
  
df_toy1=[]
for j in range(0,dim_toy2):
    for i in range(j,dim_toy2):
        df_toy1.append([str(j),str(i),toy_net2[i,j]])

df_toy1=pd.DataFrame(df_toy1)
df_toy1.columns=['source','target','weight']
    
graph_toy1=sg.StellarGraph(edges=df_toy1, is_directed=False)



rw_toy1 = BiasedRandomWalk(graph_toy1, p = 0.25, q = 1.25, n = 30, length = 250, 
                      seed=42, weighted = False)

walks_toy1 = rw_toy1.run(nodes=list(graph_toy1.nodes()))

str_walks_toy1 = [[str(n) for n in walk] for walk in walks_toy1]
    
model_toy1 = Word2Vec(str_walks_toy1, vector_size=10, window=10, min_count=1, sg=1, workers=1, epochs=1)


# Retrieve node embeddings and corresponding subjects
node_ids_toy1 = model_toy1.wv.index_to_key  # list of node IDs
node_embeddings_toy1 = (model_toy1.wv.vectors) 

    
# k means
    
distortions = []
K = range(1,25)
for k in K:
    k_cluster_toy1 = KMeans(n_clusters=k, max_iter=500, random_state=42).fit(node_embeddings_toy1)
    k_cluster_toy1.fit(node_embeddings_toy1)
    distortions.append(k_cluster_toy1.inertia_)
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for toy network - case 1')
plt.show()

# k-means dots graph for toy1

best_k_toy1=15
k_cluster_toy1 = KMeans(n_clusters=best_k_toy1, max_iter=500, random_state=42).fit(node_embeddings_toy1)
kmeans_labels_toy1 = k_cluster_toy1.labels_
nodes_labels_toy1 = pd.DataFrame(zip(node_ids_toy1, kmeans_labels_toy1),columns = ['node_ids','kmeans'])

trans_toy1 = TSNE(n_components = 2, early_exaggeration = 12,
                  perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                  learning_rate = 200.0, random_state = 74)
node_embeddings_2d_toy1 = trans_toy1.fit_transform(node_embeddings_toy1)

data_tsne_toy1 = pd.DataFrame(zip(node_ids_toy1, list(node_embeddings_2d_toy1[:,0]),list(node_embeddings_2d_toy1[:,1])),
                        columns = ['node_ids','x','y'])
data_tsne_toy1 = pd.merge(data_tsne_toy1, nodes_labels_toy1, left_on='node_ids', right_on='node_ids',
                how = 'left')

plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_toy1, x='x', y='y',hue='kmeans', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec clusters with k-means on toy-network 1')
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


## True network representation
toy1_labels=np.zeros(len(node_ids_toy1))

i=0
for node in node_ids_toy1:
    if (int(node) <= zona_b11[-1]):
        toy1_labels[i]=1 # node in the 1st block
    else:
        toy1_labels[i]=2 # node in the 2nd block
    i+=1

true_toy1_labels= pd.DataFrame(zip(node_ids_toy1, toy1_labels),columns = ['node_ids','toy1 labels']) 

data_tsne_toy1 = pd.DataFrame(zip(node_ids_toy1, list(node_embeddings_2d_toy1[:,0]),list(node_embeddings_2d_toy1[:,1])),
                        columns = ['node_ids','x','y'])
data_tsne_toy1 = pd.merge(data_tsne_toy1, true_toy1_labels, left_on='node_ids', right_on='node_ids',
                how = 'left')
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_toy1, x='x', y='y',hue='toy1 labels', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec embedding of the toy-network 1')
plt.legend(labels=['block 2', 'block 1'])
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


#%%
# Toy network analysis - CASE 2
# k-means clustering 
  
df_toy2=[]
for j in range(0,dim_toy2):
    for i in range(j,dim_toy2):
        df_toy2.append([str(j),str(i),toy_net2bis[i,j]])

df_toy2=pd.DataFrame(df_toy2)
df_toy2.columns=['source','target','weight']
    
graph_toy2=sg.StellarGraph(edges=df_toy2, is_directed=False)



rw_toy2 = BiasedRandomWalk(graph_toy2, p = 0.25, q = 1.25, n = 30, length = 250, 
                      seed=42, weighted = False)

walks_toy2 = rw_toy2.run(nodes=list(graph_toy2.nodes()))

str_walks_toy2 = [[str(n) for n in walk] for walk in walks_toy2]
    
model_toy2 = Word2Vec(str_walks_toy2, vector_size=10, window=10, min_count=1, sg=1, workers=1, epochs=1)


# Retrieve node embeddings and corresponding subjects
node_ids_toy2 = model_toy2.wv.index_to_key  # list of node IDs
node_embeddings_toy2 = (model_toy2.wv.vectors) 

    
# k means
    
distortions = []
K = range(1,25)
for k in K:
    k_cluster_toy2= KMeans(n_clusters=k, max_iter=500, random_state=12).fit(node_embeddings_toy2)
    k_cluster_toy2.fit(node_embeddings_toy2)
    distortions.append(k_cluster_toy2.inertia_)
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for toy network - case 2')
plt.show()

# k-means dots graph for toy2

best_k_toy2=15
k_cluster_toy2 = KMeans(n_clusters=best_k_toy2, max_iter=500, random_state=42).fit(node_embeddings_toy2)
kmeans_labels_toy2 = k_cluster_toy2.labels_
nodes_labels_toy2 = pd.DataFrame(zip(node_ids_toy2, kmeans_labels_toy2),columns = ['node_ids','kmeans'])

trans_toy2 = TSNE(n_components = 2, early_exaggeration = 12,
                  perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                  learning_rate = 200.0, random_state = 29)
node_embeddings_2d_toy2 = trans_toy2.fit_transform(node_embeddings_toy2)

data_tsne_toy2 = pd.DataFrame(zip(node_ids_toy2, list(node_embeddings_2d_toy2[:,0]),list(node_embeddings_2d_toy2[:,1])),
                        columns = ['node_ids','x','y'])
data_tsne_toy2 = pd.merge(data_tsne_toy2, nodes_labels_toy2, left_on='node_ids', right_on='node_ids',
                how = 'left')

plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_toy2, x='x', y='y',hue='kmeans', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec clusters with k-means on toy-network 2')
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


## True network representation
toy2_labels=np.zeros(len(node_ids_toy2))

i=0
for node in node_ids_toy2:
    if (int(node) <= zona_b12[-1]):
        toy2_labels[i]=0 # node in the 1st block
    else:
        toy2_labels[i]=1 # node in the 2nd block
    i+=1

true_toy2_labels= pd.DataFrame(zip(node_ids_toy2, toy2_labels),columns = ['node_ids','toy2 labels']) 

data_tsne_toy2 = pd.DataFrame(zip(node_ids_toy2, list(node_embeddings_2d_toy2[:,0]),list(node_embeddings_2d_toy2[:,1])),
                        columns = ['node_ids','x','y'])
data_tsne_toy2 = pd.merge(data_tsne_toy2, true_toy2_labels, left_on='node_ids', right_on='node_ids',
                how = 'left')
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_toy2, x='x', y='y',hue='toy2 labels', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec embedding of the toy-network 2')
plt.legend(labels=['block 2', 'block 1'])
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


#%%
# Loop for varying embedding dimension on toy-network 1

dim_emb=np.array([5,10,20,50,100,200,300])

for dim_e in dim_emb:
            
    model_toy1 = Word2Vec(str_walks_toy1, vector_size=dim_e, window=10, min_count=1, sg=1, workers=1, epochs=1)
    node_ids_toy1 = model_toy1.wv.index_to_key  # list of node IDs
    node_embeddings_toy1 = (model_toy1.wv.vectors) 
       
    trans_toy1 = TSNE(n_components = 2, early_exaggeration = 12,
                      perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                      learning_rate = 200.0, random_state = 74+dim_e)
    node_embeddings_2d_toy1 = trans_toy1.fit_transform(node_embeddings_toy1)
    
    ## True network representation
    toy1_labels=np.zeros(len(node_ids_toy1))
    
    i=0
    for node in node_ids_toy1:
        if (int(node) <= zona_b11[-1]):
            toy1_labels[i]=0 # node in the 1st block
        else:
            toy1_labels[i]=1 # node in the 2nd block
        i+=1
    
    true_toy1_labels= pd.DataFrame(zip(node_ids_toy1, toy1_labels),columns = ['node_ids','toy1 labels']) 
    
    data_tsne_toy1 = pd.DataFrame(zip(node_ids_toy1, list(node_embeddings_2d_toy1[:,0]),list(node_embeddings_2d_toy1[:,1])),
                            columns = ['node_ids','x','y'])
    data_tsne_toy1 = pd.merge(data_tsne_toy1, true_toy1_labels, left_on='node_ids', right_on='node_ids',
                    how = 'left')
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data_tsne_toy1, x='x', y='y',hue='toy1 labels', palette="bright",
                   alpha=0.55, s=200).set_title('Node2vec embedding of the toy-network 1, embedding dim={}'.format(dim_e))
    plt.legend(labels=['block 2', 'block 1'])
    #plt.savefig('images/kmeans_node2vec.svg')
    plt.show()
    

#%%
# Loop for varying embedding dimension on toy-network 2

dim_emb=np.array([5,10,20,50,100,200,300])

for dim_e in dim_emb:
            
    model_toy2 = Word2Vec(str_walks_toy2, vector_size=dim_e, window=10, min_count=1, sg=1, workers=1, epochs=1)
    node_ids_toy2 = model_toy2.wv.index_to_key  # list of node IDs
    node_embeddings_toy2 = (model_toy2.wv.vectors) 
       
    trans_toy2 = TSNE(n_components = 2, early_exaggeration = 12,
                      perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                      learning_rate = 200.0, random_state = 19+dim_e)
    node_embeddings_2d_toy2 = trans_toy2.fit_transform(node_embeddings_toy2)
    
    ## True network representation
    toy2_labels=np.zeros(len(node_ids_toy2))
    
    i=0
    for node in node_ids_toy2:
        if (int(node) <= zona_b12[-1]):
            toy2_labels[i]=0 # node in the 1st block
        else:
            toy2_labels[i]=1 # node in the 2nd block
        i+=1
    
    true_toy2_labels= pd.DataFrame(zip(node_ids_toy2, toy2_labels),columns = ['node_ids','toy2 labels']) 
    
    data_tsne_toy2 = pd.DataFrame(zip(node_ids_toy2, list(node_embeddings_2d_toy2[:,0]),list(node_embeddings_2d_toy2[:,1])),
                            columns = ['node_ids','x','y'])
    data_tsne_toy2 = pd.merge(data_tsne_toy2, true_toy2_labels, left_on='node_ids', right_on='node_ids',
                    how = 'left')
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data_tsne_toy2, x='x', y='y',hue='toy2 labels', palette="bright",
                   alpha=0.55, s=200).set_title('Node2vec embedding of the toy-network 2, embedding dim={}'.format(dim_e))
    plt.legend(labels=['block 2', 'block 1'])
    #plt.savefig('images/kmeans_node2vec.svg')
    plt.show()



#%%
# HI_C NETWORK NODE2VEC 



#%%
# Node2vec on Hi-C data, analysis on chr.6 and X (2nd and 3rd one)

hic_6X_list_tuples=[]


for j in range(boundary_start[1],boundary_end[2]):
    for i in range(j,boundary_end[2]):
        #if(np.log10(hic_dati[i,j]+1)> 1.8):
        if(hic_dati[i,j]!=0): 
            hic_6X_list_tuples.append((str(i), str(j), hic_dati[i,j]))
            
            
graph_6X = Graph(hic_6X_list_tuples, directed=False, weighted=True)       

start_time_6X = time.time()     

good_tries_hic = np.array([['topn', 'n_id_out_nodes', 'n_out', 'n_id_nodes_out_%']])

p_6X=0.25
q_6X=1.25
walk_6X=250 

n2v_6X = Node2Vec(graph_6X, dim=10, walk_length=walk_6X, context=300, p= p_6X, q= q_6X, workers=1)

range_topn=np.array([5,10,15,20,25,30,40,50,60], dtype=int)

for value in range_topn:
    topn=value
    
    similar_output_node2_6X=[]  
    # list as long as the number of nodes in chr.6 having outlier links
    # Each element in the list is a list of tuples: a tuple for each one of the identified most similar nodes

    
    Similar_node2_6X=[]  
    # List of lists. As many lists as the number of nodes in chr.6 having outlier links
    # For each of these nodes, there is a list of its most similar nodes 
    
    for node in hic_out_list:
        similar_output_node2_6X.append( n2v_6X.wv.most_similar(str(node),topn=topn))
          
            
    for lista in similar_output_node2_6X:   # a list of tuples for each input node
         a=[]    
         for n in lista:     # n=tuple in a list
             a.append(n[0])       # considering just most similar nodes, negletting the 2nd values in output
                
         Similar_node2_6X.append(a)
            
    #comparing results
        
    Identified_Outliers_node2_6X=[]
        
    n_id_links_out_6X=0  # number of identified links outlier
        
    for i in range(len(hic_out_list)):
        Identified_Outliers_node2_6X.append(list(set(Similar_node2_6X[i]).intersection(hic_out_list_of_lists[i])))
        n_id_links_out_6X += len(list(set(Similar_node2_6X[i]).intersection(hic_out_list_of_lists[i])))
        
    id_nodes_out_6X=[]
    for lista in Identified_Outliers_node2_6X:
        for node in lista:
            id_nodes_out_6X.append(node)
    
    id_nodes_out_6X=list(set(id_nodes_out_6X))
    
    n_id_out_nodes_6X=len(id_nodes_out_6X)  
    
    true_n_out_6X=[]
    for lista in hic_out_list_of_lists:
        for node in lista:
            true_n_out_6X.append(node)
    
    true_n_out_6X=sorted(list(set(true_n_out_6X)))
    
    n_true_n_out_6X=len(true_n_out_6X)
    
    if(n_id_out_nodes_6X !=0):
            print('Hi-C network ok con topn= {}, n_links_out_id={}, n_id_out_nodes={}, out_nodes_id_%={}'.format(topn, n_id_links_out_6X, n_id_out_nodes_6X, n_id_out_nodes_6X/n_true_n_out_6X*100) )
            good_tries_hic= np.append(good_tries_hic, [[int(topn), int(n_id_out_nodes_6X), int(n_true_n_out_6X), int(n_id_out_nodes_6X/n_true_n_out_6X*100)]], axis=0)    


good_tries_hic = np.delete(good_tries_hic, (0), axis=0) 

good_tries_hic= pd.DataFrame(good_tries_hic,columns= [ 'topn', 'n_id_out_nodes', 'n_out', 'n_id_nodes_out_%'] )

good_rows_hic=good_tries_hic.index[good_tries_hic['n_id_out_nodes'] == max(good_tries_hic['n_id_out_nodes'])].tolist()        


print("Max number of identified outlier nodes: {} out of {}".format(max(good_tries_hic['n_id_out_nodes']), n_true_n_out_6X))

print("--- %s seconds ---" % (time.time() - start_time_6X))


#%%
#Plot n_id_out VS topn Hi-C network

plt.plot(good_tries_hic['topn'],good_tries_hic['n_id_nodes_out_%'], 'ro')
plt.xlim=(0,max(range_topn))
plt.ylim=(0,100)
plt.xlabel('topn value')
plt.ylabel('Nodes outliers identified (%)')
plt.title('Hi-C data')
plt.grid(True)
plt.show()

#%%
# Here manual setting of the desired topn value in most similar function (Hi-C data)

graph_6X = Graph(hic_6X_list_tuples, directed=False, weighted=True)   
p_6X=0.25
q_6X=1.25
walk_6X=250 

n2v_6X = Node2Vec(graph_6X, dim=10, walk_length=walk_6X, context=300, p= p_6X, q= q_6X, workers=1)

similar_output_node2_6X=[]  

Similar_node2_6X=[]  
      
for node in hic_out_list:
    similar_output_node2_6X.append( n2v_6X.wv.most_similar(str(node),topn=20))
          
            
for lista in similar_output_node2_6X: 
     a=[]    
     for n in lista:    
         a.append(n[0])        
                
     Similar_node2_6X.append(a)
            
#comparing results
      
Identified_Outliers_node2_6X=[]

for i in range(len(hic_out_list)):
    Identified_Outliers_node2_6X.append(list(set(Similar_node2_6X[i]).intersection(hic_out_list_of_lists[i])))
        
id_nodes_out_6X=[]
for lista in Identified_Outliers_node2_6X:
    for node in lista:
        id_nodes_out_6X.append(node)
    
id_nodes_out_6X=list(set(id_nodes_out_6X))
    
n_id_out_nodes_6X=len(id_nodes_out_6X)  
    
true_n_out_6X=[]
for lista in hic_out_list_of_lists:
    for node in lista:
        true_n_out_6X.append(node)
    
true_n_out_6X=sorted(list(set(true_n_out_6X)))
    
n_true_n_out_6X=len(true_n_out_6X)
    
if(n_id_out_nodes_6X !=0):
   print('Hi-C network topn= 20: n_id_out_nodes={}, out_nodes_id_%={}'.format(n_id_out_nodes_6X, n_id_out_nodes_6X/n_true_n_out_6X*100) )

nodes_out_not_found=sorted(list(set(true_n_out_6X)-set(id_nodes_out_6X)))   



#%%
# Suggesting in input nodes with no outlier links

ok_similar_output_node2_6X=[]  
 
ok_Similar_node2_6X=[]  

for node in range(boundary_end[1]-55,boundary_end[1]):      
    ok_similar_output_node2_6X.append( n2v_6X.wv.most_similar(str(node),topn=20)) 
          
            
for lista in ok_similar_output_node2_6X:          
    a=[]    
    for n in lista:    
        a.append(n[0])       
    ok_Similar_node2_6X.append(a)

#Results
     
ok_Similar_Outliers_node2_6X=[]

for i in range(55):
    ok_Similar_Outliers_node2_6X.append(list(set(ok_Similar_node2_6X[i]).intersection(true_n_out_6X)))
        
                  
ok_sim_nodes_out=[]
for lista in ok_Similar_Outliers_node2_6X:
    for node in lista:
       ok_sim_nodes_out.append(node)

    
ok_sim_nodes_out=sorted(list(set(ok_sim_nodes_out)))
    
ok_n_sim_out_nodes=len(ok_sim_nodes_out)  

result_check=len(list(set(ok_sim_nodes_out).intersection(true_n_out_6X)))*100/len(true_n_out_6X)

#%%
# Graphs of identified outliers per node suggested


# Nodes with outlier links in input
x_out=list(int(node) for node in hic_out_list)
y_out=[]
for i in range(55):
    y_out.append(len(Identified_Outliers_node2_6X[i]))

plt.bar(x_out,y_out)
plt.xlabel('Input node index')
plt.ylabel('# of identified outliers (out of 56)')
plt.title('Outliers in input, topn=20, p=0.25, q=1.25, walk=250')
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(axis='y')
plt.show()



plt.bar(*np.unique(y_out, return_counts=True))
plt.xlabel('# of identified outliers per node suggested')
plt.title('Outliers in input, topn=20, p=0.25, q=1.25, walk=250')
plt.grid(axis='y')
plt.show()


#####
# Nodes with no outlier links in input
x_ok=np.arange(boundary_end[1]-55,boundary_end[1])
y_ok=[]
for i in range(55):
    y_ok.append(len(ok_Similar_Outliers_node2_6X[i]))

plt.bar(x_ok,y_ok)
plt.xlabel('Input node index')
plt.ylabel('# of identified outliers (out of 56)')
plt.title('Non-outliers in input, topn=20, p=0.25, q=1.25, walk=250')
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(np.arange(10,dtype=int))
plt.grid(axis='y')
plt.show()


plt.bar(*np.unique(y_ok, return_counts=True))
plt.xlabel('# of identified outliers per node suggested')
plt.title('Non-outliers in input, topn=20, p=0.25, q=1.25, walk=250')
plt.grid(axis='y')
plt.show()

#%%
# Visualization network 6X


# For color mapping
import matplotlib.colors as colors
import matplotlib.cm as cmx

# Group (chromosome) of belonging

hic_6X_df=pd.DataFrame({'Node id':list(zona_6X),'Group':list(np.zeros(len(zona_6X)))})  

val_map = {}
keys = range(len(zona_6X))

for n in range(0,len(zona_6X)):
    if (str(hic_6X_df['Node id'][n]) in hic_out_list):
        hic_6X_df['Group'][n]= 's_out' # suggested (true) outlier 
        val_map[n]=1
    elif(str(hic_6X_df['Node id'][n]) in id_nodes_out_6X):
         hic_6X_df['Group'][n]= 'id_out' # identified (true) outlier
         val_map[n]=2
    #elif(str(hic_6X_df['Node id'][n]) in nodes_out_not_found):
     #    hic_6X_df['Group'][n]= 'out_not_found'
      #   val_map[n]=5
    elif(int(hic_6X_df['Node id'][n]) < boundary_start[2] and str(hic_6X_df['Node id'][n]) not in hic_out_list):
        hic_6X_df['Group'][n]= 'chr6' # belonging to chr6
        val_map[n]=3
    else:
        hic_6X_df['Group'][n]= 'chrX' # belonging to chrX
        val_map[n]=4
    
    
#ColorLegend = {'Suggested outliers': 1,'Chr.6': 3,'Identified outliers': 2,'Chr.X': 4, 'Not identified outliers': 5}
ColorLegend = {'Suggested outliers': 1,'Chr.6': 3,'Identified outliers': 2,'Chr.X': 4}

       
# Edge list

hic_6X_edges=pd.DataFrame()
from_6X=[]
to_6X=[]


for j in range(boundary_start[1],boundary_end[2]):
    for i in range(j,boundary_end[2]):
        if(np.log10(hic_dati[i,j]+1) > 1.8):
        #if(hic_dati[i,j]!=0): 
            from_6X.append(j)
            to_6X.append(i)

hic_6X_edges.insert(0,'from',from_6X,True)
hic_6X_edges.insert(1,'to',to_6X,True)

# graph 
G=nx.from_pandas_edgelist(hic_6X_edges, 'from', 'to', create_using=nx.Graph() )

pos=nx.spring_layout(G,k=0.85)

values = [val_map.get(node-250, 0) for node in G.nodes()]
# Color mapping
jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=max(values))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# Using a figure to use it as a parameter when calling nx.draw_networkx
f = plt.figure(1)
ax = f.add_subplot(1,1,1)
for label in ColorLegend:
    ax.plot([0],[0],color=scalarMap.to_rgba(ColorLegend[label]),label=label)


# Custom the nodes:
nx.draw(G, pos, cmap = jet, vmin=0, vmax= max(values),node_color=values, with_labels=False, node_size=25, linewidths=1, width=0.02)                                                                                                       
plt.axis('off')
f.set_facecolor('w')

plt.legend()

f.tight_layout()
plt.show()


#%%
# Measure of the degree (strength) of nodes out not found 
    
w_degree=[]
for node in nodes_out_not_found:
    w_degree.append(G.degree(weight='weight')[int(node)])

mean_deg_out=np.mean(w_degree) # average degree of out not found

mean_deg= np.mean(G.degree(weight='weight')) # average degree of the network

w_deg_id=[]
for node in id_nodes_out_6X:
    w_deg_id.append(G.degree(weight='weight')[int(node)])
    
mean_deg_id=np.mean(w_deg_id) # average degree of out identified

## NB: the degree changes as the number of edges in the network varies


#%%
# K-means clustering on dati Hi-C (analysis on chromosomes 6 and X, the 2nd and 3rd one)

zona_6X = np.arange(boundary_start[1],boundary_end[2], dtype=int)
net_6X = hic_dati[np.ix_(zona_6X,zona_6X)]

dim_6X= len_chrom[1]+len_chrom[2]
  
df_6X=[]
for j in range(0, dim_6X):
    for i in range(j, dim_6X):
        df_6X.append([str(j),str(i),net_6X[i,j]])

df_6X=pd.DataFrame(df_6X)
df_6X.columns=['source','target','weight']
    
graph_6X=sg.StellarGraph(edges=df_6X, is_directed=False)

#Generation of random walks:
    
rw_6X = BiasedRandomWalk(graph_6X, p = 0.25, q = 1.25, n = 30, length = 250, 
                      seed=11, weighted = True)

walks_6X = rw_6X.run(nodes=list(graph_6X.nodes())
               # root nodes
              )

# we pass the random walks to a list
str_walks_6X = [[str(n) for n in walk] for walk in walks_6X]
    
model_6X = Word2Vec(str_walks_6X, vector_size=10, window=30, min_count=1, sg=1, workers=1, epochs=1)
#vector_size is the number of features the code look for
   
model_6X.wv.save_word2vec_format("EMBEDDING_6X")
# Save model for later use
model_6X.save("EMBEDDING_MODEL_6X")
# Retrieve node embeddings and corresponding subjects
node_ids_6X = model_6X.wv.index_to_key  # list of node IDs
node_embeddings_6X = (model_6X.wv.vectors) 

   
# k means elbow for net_6X
    
distortions_6X = []
K = range(1,25)
for k in K:
    k_cluster_6X = KMeans(n_clusters=k, max_iter=500, random_state=11).fit(node_embeddings_6X)
    k_cluster_6X.fit(node_embeddings_6X)
    distortions_6X.append(k_cluster_6X.inertia_)
    
    
# Plot the elbow
plt.plot(K, distortions_6X, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for Hi-C data')
plt.show()

#%%
# k-means dots graph for Hi-C

best_k=15
k_cluster_6X = KMeans(n_clusters=best_k, max_iter=500, random_state=11).fit(node_embeddings_6X)
kmeans_labels_6X = k_cluster_6X.labels_
nodes_labels_6X = pd.DataFrame(zip(node_ids_6X, kmeans_labels_6X),columns = ['node_ids_6X','kmeans'])


# fit  embeddings with t-SNE
from sklearn.manifold import TSNE
trans_6X = TSNE(n_components = 2, early_exaggeration = 12,
                  perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                  learning_rate = 200.0, random_state = 11)
node_embeddings_2d_6X = trans_6X.fit_transform(node_embeddings_6X)

# create the dataframe that has information about the nodes and their x and y coordinates
data_tsne_6X = pd.DataFrame(zip(node_ids_6X, list(node_embeddings_2d_6X[:,0]),list(node_embeddings_2d_6X[:,1])),
                        columns = ['node_ids_6X','x','y'])
data_tsne_6X = pd.merge(data_tsne_6X, nodes_labels_6X, left_on='node_ids_6X', right_on='node_ids_6X',
                how = 'left')

import seaborn as sns
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_6X, x='x', y='y',hue='kmeans', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec clusters with k-means on Hi-C data')
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


# True Hi-C network representation

i=0
labels_6X= np.zeros(len(node_ids_6X))
for node in node_ids_6X:
    if (str(int(node)+boundary_start[1]) in hic_out_list):
        labels_6X[i]=0 # node in chr.6 with outlier links 
    elif(str(int(node)+boundary_start[1]) in true_n_out_6X):
         labels_6X[i]=1 # node in chr.X with outlier links
    elif((int(node)+boundary_start[1]) < boundary_start[2] and str(int(node)+boundary_start[1]) not in hic_out_list):
       labels_6X[i]=2 # node in chr6 with no outlier links
    else:
        labels_6X[i]=3 # node in chr.X with no outlier links
    i+=1
    
true_labels_6X = pd.DataFrame(zip(node_ids_6X, labels_6X),columns = ['node_ids_6X','6X labels'])

data_tsne_6X_true = pd.DataFrame(zip(node_ids_6X, list(node_embeddings_2d_6X[:,0]),list(node_embeddings_2d_6X[:,1])),
                        columns = ['node_ids_6X','x','y'])
data_tsne_6X_true = pd.merge(data_tsne_6X_true, true_labels_6X, left_on='node_ids_6X', right_on='node_ids_6X',
                how = 'left')

plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_6X_true, x='x', y='y',hue='6X labels', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec embedding of Hi-C data network')
plt.legend(labels=['chr.X', 'out chr.6', 'out chr.X', 'chr.6'])
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


#%%
# t-SNE plots as dim embedding varies - Hi-C data

dim_emb=np.array([5,10,20,50,100,200,300])

for dim_e in dim_emb:
            
    model_6X = Word2Vec(str_walks_6X, vector_size=10, window=10, min_count=1, sg=1, workers=1, epochs=1)
    node_ids_6X = model_6X.wv.index_to_key  # list of node IDs
    node_embeddings_6X = (model_6X.wv.vectors) 
       
    trans_6X = TSNE(n_components = 2, early_exaggeration = 12,
                      perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                      learning_rate = 200.0, random_state = 123+dim_e)
    node_embeddings_2d_6X = trans_6X.fit_transform(node_embeddings_6X)
    
    
    i=0
    labels_6X= np.zeros(len(node_ids_6X))
    for node in node_ids_6X:
        if (str(int(node)+boundary_start[1]) in hic_out_list):
            labels_6X[i]=0 # node in chr.6 with outlier links 
        elif(str(int(node)+boundary_start[1]) in true_n_out_6X):
             labels_6X[i]=1 # node in chr.X with outlier links
        elif((int(node)+boundary_start[1]) < boundary_start[2] and str(int(node)+boundary_start[1]) not in hic_out_list):
           labels_6X[i]=2 # node in chr6 with no outlier links
        else:
            labels_6X[i]=3 # node in chr.X with no outlier links
        i+=1
        
    true_labels_6X = pd.DataFrame(zip(node_ids_6X, labels_6X),columns = ['node_ids_6X','6X labels'])
    
    data_tsne_6X_true = pd.DataFrame(zip(node_ids_6X, list(node_embeddings_2d_6X[:,0]),list(node_embeddings_2d_6X[:,1])),
                            columns = ['node_ids_6X','x','y'])
    data_tsne_6X_true = pd.merge(data_tsne_6X_true, true_labels_6X, left_on='node_ids_6X', right_on='node_ids_6X',
                    how = 'left')
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data_tsne_6X_true, x='x', y='y',hue='6X labels', palette="bright",
                   alpha=0.55, s=200).set_title('Node2vec embedding of Hi-C data, embedding dim={}'.format(dim_e))
    plt.legend(labels=['chr.X', 'out chr.6', 'out chr.X', 'chr.6'])
    #plt.savefig('images/kmeans_node2vec.svg')
    plt.show()
    
    
#%%
# Saving the embedding vectors for p=0.25, q=1.25, walk length=250, dim=10. 
# Each row of the matrix will correspond to a node: the first value is the node index, 
#  the other values on the row are the corresponding vector's elements 

graph_6X = Graph(hic_6X_list_tuples, directed=False, weighted=True)   
p_6X=0.25
q_6X=1.25
walk_6X=250 

n2v_6X = Node2Vec(graph_6X, dim=10, walk_length=walk_6X, context=300, p= p_6X, q= q_6X, workers=1)

emb_vectors_6X=[]
key_err=[]


for i in zona_6X:
   try:
       emb_vectors_6X.append([str(i), n2v_6X.wv[str(i)]]) 
   except KeyError:
       key_err.append(i)
       
emb_array=np.zeros((len(emb_vectors_6X),10))
emb_labels=[]

i=0
for list in emb_vectors_6X:
    emb_labels.append(list[0])       
    emb_array[i,:]=list[1]
    i+=1

df_emb_vectors_6X=pd.DataFrame(emb_array, index=emb_labels)

#df_emb_vectors_6X.to_csv(r'...path...\emb_vectors_6X.txt', sep=' ', index=True, header=False)

