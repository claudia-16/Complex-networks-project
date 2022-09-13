# Simulation of chromosome contact maps and analysis with node2vec algorithm
This project focuses on the generation of two "toy-networks" mimicking chromosome contact maps. These simplfied networks are based on Erdos-Renyi networks and
the purpose of the project is to analyse these matrixes by using the node2vec algorithm. This algorithm will provide a node embedding on which we can work and 
see how nodes can be labeled. For the labelling task we focused here on the K-means clustering algorithm and we compare its output with the true labels. 
Further checks are made by changing the embedding dimension set in the node2vec algorithm. All the comparisons here are made via t-SNE visualization.

# Project structure
***1) Networks generation*** \
The two networks generated are block-networks intended to represent a simplfied version of chromosomes contact maps. We considered the presence of two chromosomes.
Matrixes are  therefore symmetric matrixes, whose indexes represent genomic loci (which are the nodes of the network). In this simplified representation, 
matrixes are binary, having 1 for indicating the presence of a link and 0 for its absence.
The majority of interactions (i.e. links) occurs within chromosomes, therefore matrixes are block matrixes having the majority of non-zero values 
in blocks located along the main diagonal.
We considered each chromosome block as an Erdos-Renyi network having a link probability given as input parameter. The off-diagonal blocks come too from
an Erdos-Renyi network in which the link probability is given by the noise. 
Further parameters we set are the chromosome dimensions, i.e the block sizes. The networks generated are two:
in the first one we consider chromosomes of the same size (both of them have 200 loci); in the second network instead we have chromosomes of very different sizes 
(one has 80 loci and the other 320). Overall the two networks have the same dimensions of 400 nodes.

***2) Network analysis*** \
For the two networks we adopted same analysis approach. 
First, we used node2vec algorithm to obtain a node embedding. The node2vec algorithm generates a series of pseudo-random walks through the network.
The way these walks are biased can be set by the user modifying the "p" and "q" parameters, choosing a more BFS-like approach rather than one more DFS-like
(see for details *node2vec: Scalable Feature Learning for Networks, 2016. url: https://arxiv.org/abs/1607.00653*).
These walks are sequences of nodes from which a node embedding is extracted.
Once we obtain the node embedding, we apply to it the K-means clustering algorithm. The optimal value for k is decided by using the elbow method.
We compare then the way nodes are labelled by the k-means algorithm with the true labels we know (i.e., the chromosome of belonging). This comparison is made visually by using a t-SNE algorithm. We make also further analysis by changing the embedding dimension parameter of node2vec and seeing how nodes, with their true labels, are located in the t-SNE visualization.

# Dependencies
- random
- numpy
- pandas
- seaborn
- matplotlib.pyplot
- gensim.models
- stellargraph.data
- sklearn.cluster
- sklearn.manfold

# Main parameters
In the following lines are indicated the main parameters the user can vary. \
In the networks generation part:
- *DIM_TOY2*: overall dimension of the "toy-network" generated;
- *P_NOISE*: probability of having a noise link in the network;
- *P_BLOCKS*: probability of having a link between 2 loci belonging to the same chromosome;
- *DIM_BLOCKS*: size of the chromosomes for the case in which the two of them have equal size. \
(N.B.: this parameter should be coherent with *DIM_TOY2*);
- *DIM_BLOCK1* and *DIM_BLOCK2*: sizes of the two chromosomes (for the case in which their sizes can be different). \
(N.B.: their values should be coherent with *DIM_TOY2*).

In the network analysis part:
- *p*, *q*, *n*, *length*: parameters of BiasedRandomWalk related to the generation of the biased random walks through the network;
- *vector_size*, *window*, *min_count*, *sg*, *workers and *epochs*: parameters of Word2Vec;
- *K*: range of k values tested with the k-means clustering algorithm to obtain the elbow curve;
- *BEST_K_TOY1* and *BEST_K_TOY2*: optimal k values for the two networks individuated by looking at the elbow plot;
- *early_exaggeration*, *perplexity*, *n_iter*, *n_iter_without_progress* and *learning_rate*: parameters of the t-SNE algorithm;
- *dim_emb*: array of the embedding dimensions tested.
