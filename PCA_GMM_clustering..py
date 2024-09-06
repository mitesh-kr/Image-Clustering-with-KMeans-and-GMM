#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load Dataset
data_train = pd.read_csv('/content/mnist_train.csv')

#Seperate Label and Pixel
data_train_label = data_train.iloc[:,:1]
data_train_pixel = data_train.iloc[:,1:]

#Normalization
data = data_train_pixel/255
data = np.array(data)

#PCA
def pca(n_components):
    U,S,Vt = np.linalg.svd(data, full_matrices=False)
    pca_data = np.dot(data,Vt.T[:,:n_components])
    return pca_data

#GMM
def gmm(k_clusters,data):
    gmm = GaussianMixture(n_components=k_clusters,max_iter=500)
    gmm.fit(data)
    cluster_labels = gmm.predict(data)
    return cluster_labels


# creating a loop for execution of for 128,64, and 32 components and 10,7, and 4 clusters.
for n_components in [128,64,32]:
    for k_clusters in [10,7,4]:
        principle_data = pca(n_components)
        cluster_labels = gmm(k_clusters,principle_data)

        # Plot the clustered data points
        plt.scatter(principle_data[:, 0],principle_data[:, 1], c=cluster_labels, s=10, cmap='viridis')
        plt.title('{} components {} clusters '.format(n_components,k_clusters))
        plt.show()
        print('Total {} Clusters'.format(k_clusters))
        # Image visulals
        for x in range (k_clusters):
            clustere_data = data[cluster_labels==x]
            sample = np.random.choice(len(clustere_data), size=10, replace=False)
            print('samples from cluster {} out of total'.format(x+1), len(clustere_data),'datapoint')
            plt.figure(figsize=(8, 8))
            for j, idx in enumerate(sample):
                plt.subplot(1, 10, j + 1)
                plt.imshow(clustere_data[idx].reshape(28,28), cmap='gray')
                plt.axis('off')
            plt.show()

