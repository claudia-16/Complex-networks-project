# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:51:38 2022

@author: Claudia Ghinato
"""

# Software & Computing for Applied Physics project

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import stellargraph as sg
from stellargraph.data import BiasedRandomWalk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


SEED=18
random.seed(SEED)

#%%
# Useful functions

def triupper(prob, matrix):

    ''''Function for obtaining an upper triangular matrix (0s on the main diagonal).
        The values above the main diagonal are randomly 0 or 1, according to the probability "prob".
        The function overwrites the input matrix

        Parameters
        ----------
        prob : positive float
            Probability of having a link (value 1) in the upper triangular matrix
        matrix : numpy ndarray
            Input square matrix to be transformed
        Returns
        -------
        numpy ndarray
        '''

    if prob<0:
        raise ValueError("Probability must be >=0")
    if prob>1:
        raise ValueError("Probability must be <=1")
    row= matrix.shape[0] # number of rows
    col=matrix.shape[1]  # number of columns
    if row!=col:
        raise IndexError("The input matrix must be a square matrix")
    if not isinstance(matrix, np.ndarray):
        raise AttributeError("The input matrix must be a numpy ndarray")
    if matrix.ndim!=2:
        raise IndexError("The input matrix must be a 2-dimensional numpy array")
    for i in range(0,row):
        for j in range(0,col):
            if (i>j or i==j):
                matrix[i][j]=0
            else:
                # probability of having a link is greater than not having it
                if random.uniform(0,1)> 1-prob:
                    matrix[i][j]=1
                else:
                    matrix[i][j]=0


def symm_block(dim,prob):
    '''
    The function produces a binary symmetric matrix having dimension given by the input value.
    The 1 values in the matrix are determined by the probability given in input.
    This function relies on triupper function.

    Parameters
    ----------
    dim : int
        Dimension of the matrix
    prob : float
        Probability that a matrix element has value 1

    Returns
    -------
    block : numpy array of int values

    '''
    aux=np.zeros((dim,dim), dtype=int)
    triupper(prob,aux)
    block=np.add(aux, np.transpose(aux))
    return block


#%%
# Simple toy networks

#CASE 1 - same size blocks

DIM_TOY2=400
toy_net2= np.zeros((DIM_TOY2,DIM_TOY2), dtype=int)

P_NOISE=0.3
P_BLOCKS=0.85

DIM_BLOCKS=200
#block 1 case 1
block11=symm_block(DIM_BLOCKS, P_BLOCKS)

#block 2 case 1
block21=symm_block(DIM_BLOCKS, P_BLOCKS)

#noise directly in the network
toy_net2=symm_block(DIM_TOY2,P_NOISE)

zona_b11= np.arange(0, DIM_BLOCKS,dtype=int)
zona_b21 = np.arange(DIM_BLOCKS, DIM_TOY2,dtype=int)

toy_net2[np.ix_(zona_b11, zona_b11)]= block11
toy_net2[np.ix_(zona_b21, zona_b21)]= block21

plt.imshow(toy_net2)
#plt.colorbar()
plt.show()

#%%

#CASE 2 - blocks of sizes very different

toy_net2bis= np.zeros((DIM_TOY2, DIM_TOY2), dtype=int)

DIM_BLOCK1=80
DIM_BLOCK2=320

# block 1 case 2
block12=symm_block(DIM_BLOCK1, P_BLOCKS)

#block 2 case 2
block22=symm_block(DIM_BLOCK2, P_BLOCKS)

#block noise directly in the network
toy_net2bis=symm_block(DIM_TOY2, P_NOISE)

zona_b12= np.arange(0, DIM_BLOCK1,dtype=int)
zona_b22 = np.arange(DIM_BLOCK1, DIM_TOY2,dtype=int)

toy_net2bis[np.ix_(zona_b12, zona_b12)]= block12
toy_net2bis[np.ix_(zona_b22, zona_b22)]= block22


plt.imshow(toy_net2bis)
#plt.colorbar()
plt.show()


#%%
# Toy network analysis - CASE 1
# k-means clustering

df_toy1=[]
for j in range(0,DIM_TOY2):
    for i in range(j,DIM_TOY2):
        df_toy1.append([str(j),str(i),toy_net2[i,j]])

df_toy1=pd.DataFrame(df_toy1)
df_toy1.columns=['source','target','weight']

graph_toy1=sg.StellarGraph(edges=df_toy1, is_directed=False)



rw_toy1 = BiasedRandomWalk(graph_toy1, p = 0.25, q = 1.25, n = 30, length = 250,
                      seed=42, weighted = False)

walks_toy1 = rw_toy1.run(nodes=list(graph_toy1.nodes()))

str_walks_toy1 = [[str(n) for n in walk] for walk in walks_toy1]

model_toy1 = Word2Vec(str_walks_toy1, vector_size=10,
                      window=10, min_count=1, sg=1, workers=1, epochs=1)


# Retrieve node embeddings and corresponding subjects
node_ids_toy1 = model_toy1.wv.index_to_key  #list of node IDs
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

BEST_K_TOY1=15
k_cluster_toy1 = KMeans(n_clusters=BEST_K_TOY1, max_iter=500,
                        random_state=42).fit(node_embeddings_toy1)
kmeans_labels_toy1 = k_cluster_toy1.labels_
nodes_labels_toy1 = pd.DataFrame(zip(node_ids_toy1, kmeans_labels_toy1),
                                 columns = ['node_ids','kmeans'])

trans_toy1 = TSNE(n_components = 2, early_exaggeration = 12,
                  perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                  learning_rate = 200.0, random_state = 74)
node_embeddings_2d_toy1 = trans_toy1.fit_transform(node_embeddings_toy1)

data_tsne_toy1 = pd.DataFrame(zip(node_ids_toy1, list(node_embeddings_2d_toy1[:,0]),
                                  list(node_embeddings_2d_toy1[:,1])),
                                  columns = ['node_ids','x','y'])
data_tsne_toy1 = pd.merge(data_tsne_toy1, nodes_labels_toy1, left_on='node_ids',
                          right_on='node_ids', how = 'left')

plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_toy1, x='x', y='y',hue='kmeans', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec clusters with k-means on toy-network 1')
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


## True network representation

toy1_labels=np.zeros(len(node_ids_toy1))

i=0
for node in node_ids_toy1:
    if int(node) <= zona_b11[-1]:
        toy1_labels[i]=1 # node in the 1st block
    else:
        toy1_labels[i]=2 # node in the 2nd block
    i+=1

true_toy1_labels= pd.DataFrame(zip(node_ids_toy1, toy1_labels),columns = ['node_ids','toy1 labels'])

data_tsne_toy1 = pd.DataFrame(zip(node_ids_toy1, list(node_embeddings_2d_toy1[:,0]),
                              list(node_embeddings_2d_toy1[:,1])), columns = ['node_ids','x','y'])
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
for j in range(0,DIM_TOY2):
    for i in range(j,DIM_TOY2):
        df_toy2.append([str(j),str(i),toy_net2bis[i,j]])

df_toy2=pd.DataFrame(df_toy2)
df_toy2.columns=['source','target','weight']

graph_toy2=sg.StellarGraph(edges=df_toy2, is_directed=False)



rw_toy2 = BiasedRandomWalk(graph_toy2, p = 0.25, q = 1.25, n = 30, length = 250,
                      seed=42, weighted = False)

walks_toy2 = rw_toy2.run(nodes=list(graph_toy2.nodes()))

str_walks_toy2 = [[str(n) for n in walk] for walk in walks_toy2]

model_toy2 = Word2Vec(str_walks_toy2, vector_size=10, window=10,
                      min_count=1, sg=1, workers=1, epochs=1)


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

BEST_K_TOY2=15
k_cluster_toy2 = KMeans(n_clusters=BEST_K_TOY2, max_iter=500,
                        random_state=42).fit(node_embeddings_toy2)
kmeans_labels_toy2 = k_cluster_toy2.labels_
nodes_labels_toy2 = pd.DataFrame(zip(node_ids_toy2,
                                     kmeans_labels_toy2),columns = ['node_ids','kmeans'])

trans_toy2 = TSNE(n_components = 2, early_exaggeration = 12,
                  perplexity = 35, n_iter = 1000, n_iter_without_progress = 500,
                  learning_rate = 200.0, random_state = 29)
node_embeddings_2d_toy2 = trans_toy2.fit_transform(node_embeddings_toy2)

data_tsne_toy2 = pd.DataFrame(zip(node_ids_toy2, list(node_embeddings_2d_toy2[:,0]),
                              list(node_embeddings_2d_toy2[:,1])), columns = ['node_ids','x','y'])
data_tsne_toy2 = pd.merge(data_tsne_toy2, nodes_labels_toy2, left_on='node_ids',
                          right_on='node_ids', how = 'left')

plt.figure(figsize=(10, 10))
sns.scatterplot(data=data_tsne_toy2, x='x', y='y',hue='kmeans', palette="bright",
               alpha=0.55, s=200).set_title('Node2vec clusters with k-means on toy-network 2')
#plt.savefig('images/kmeans_node2vec.svg')
plt.show()


## True network representation
toy2_labels=np.zeros(len(node_ids_toy2))

i=0
for node in node_ids_toy2:
    if int(node) <= zona_b12[-1]:
        toy2_labels[i]=0 # node in the 1st block
    else:
        toy2_labels[i]=1 # node in the 2nd block
    i+=1

true_toy2_labels= pd.DataFrame(zip(node_ids_toy2, toy2_labels),columns = ['node_ids','toy2 labels'])

data_tsne_toy2 = pd.DataFrame(zip(node_ids_toy2, list(node_embeddings_2d_toy2[:,0]),
                              list(node_embeddings_2d_toy2[:,1])), columns = ['node_ids','x','y'])
data_tsne_toy2 = pd.merge(data_tsne_toy2, true_toy2_labels, left_on='node_ids',
                           right_on='node_ids', how = 'left')
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

    model_toy1 = Word2Vec(str_walks_toy1, vector_size=dim_e, window=10,
                          min_count=1, sg=1, workers=1, epochs=1)
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
        if int(node) <= zona_b11[-1]:
            toy1_labels[i]=0 # node in the 1st block
        else:
            toy1_labels[i]=1 # node in the 2nd block
        i+=1

    true_toy1_labels= pd.DataFrame(zip(node_ids_toy1, toy1_labels),
                                   columns = ['node_ids','toy1 labels'])

    data_tsne_toy1 = pd.DataFrame(zip(node_ids_toy1, list(node_embeddings_2d_toy1[:,0]),
                                  list(node_embeddings_2d_toy1[:,1])),
                                  columns = ['node_ids','x','y'])
    data_tsne_toy1 = pd.merge(data_tsne_toy1, true_toy1_labels, left_on='node_ids',
                              right_on='node_ids', how = 'left')
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data_tsne_toy1, x='x', y='y',hue='toy1 labels', palette="bright",
                   alpha=0.55, s=200).set_title(
                   '''Node2vec embedding of the toy-network 1, embedding dim={}'''.format(dim_e))
    plt.legend(labels=['block 2', 'block 1'])
    #plt.savefig('images/kmeans_node2vec.svg')
    plt.show()


#%%
# Loop for varying embedding dimension on toy-network 2

dim_emb=np.array([5,10,20,50,100,200,300])

for dim_e in dim_emb:

    model_toy2 = Word2Vec(str_walks_toy2, vector_size=dim_e, window=10,
                          min_count=1, sg=1, workers=1, epochs=1)
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
        if int(node) <= zona_b12[-1]:
            toy2_labels[i]=0 # node in the 1st block
        else:
            toy2_labels[i]=1 # node in the 2nd block
        i+=1

    true_toy2_labels= pd.DataFrame(zip(node_ids_toy2, toy2_labels),
                                       columns = ['node_ids','toy2 labels'])

    data_tsne_toy2 = pd.DataFrame(zip(node_ids_toy2, list(node_embeddings_2d_toy2[:,0]),
                                      list(node_embeddings_2d_toy2[:,1])),
                                      columns = ['node_ids','x','y'])
    data_tsne_toy2 = pd.merge(data_tsne_toy2, true_toy2_labels, left_on='node_ids',
                              right_on='node_ids', how = 'left')
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data_tsne_toy2, x='x', y='y',hue='toy2 labels',palette="bright",
                    alpha=0.55, s=200).set_title('''Node2vec embedding of the toy-network 2,
                                                 embedding dim={}'''.format(dim_e))
    plt.legend(labels=['block 2', 'block 1'])
    #plt.savefig('images/kmeans_node2vec.svg')
    plt.show()
