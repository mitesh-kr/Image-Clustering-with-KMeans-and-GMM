#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
data_train = pd.read_csv('/content/mnist_train.csv')

#Seperate Label and Pixel
data_train_label = data_train.iloc[:,:1]
data_train_pixel = data_train.iloc[:,1:]

#Normalization
data = data_train_pixel/255
data = np.array(data)

# Create KMeans class
class KMeans:
    def __init__(self, k_clusters, max_iterations=1000):
        self.k_clusters = k_clusters
        self.max_iterations = max_iterations

    # Randomly initialize cluster centroids
    def initialize_centroids(self, data):
        indices = np.random.choice(len(data), size=self.k_clusters, replace=False)
        centroids = data[indices]
        return centroids

     # Compute cosine similarity between two vectors
    def cosine_similarity(self, x, y):
        similarity = (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        return similarity

    # Assign each data point to the nearest cluster by maximizing cosine similarity
    def assign_clusters(self, data, centroids):
        similarities = np.array([self.cosine_similarity(data, centroid) for centroid in centroids])
        cluster_label = np.argmax(similarities, axis=0)
        return cluster_label

    # Update cluster centroids based on the mean of data points in each cluster
    def update_centroids(self, data, cluster_label):
        centroids = np.array([data[cluster_label == k].mean(axis=0) for k in range(self.k_clusters)])
        return centroids

    #Fit datset into model
    def fit(self, data):
        centroids = self.initialize_centroids(data)
        for _ in range(self.max_iterations):
            cluster_label = self.assign_clusters(data, centroids)
            new_centroids = self.update_centroids(data, cluster_label)
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return centroids, cluster_label

# Loop for fit and visualise image for 10,7 and 4 clusters.
for cluster_number in [10,7,4]:
    print('Total {} Clusters'.format(cluster_number))
    # call KMeans
    kmeans = KMeans(k_clusters=cluster_number)
    # Fit data
    centroids, labeled_cluster= kmeans.fit(data)
    # Image visulals
    for x in range (cluster_number):
        clustere_data = data[labeled_cluster==x]
        sample = np.random.choice(len(clustere_data), size=10, replace=False)
        print('samples from cluster {} out of total'.format(x+1), len(clustere_data),'datapoint')
        plt.figure(figsize=(8,4))
        for j, idx in enumerate(sample):
            plt.subplot(1, 10, j + 1)
            plt.imshow(clustere_data[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')

        plt.show()

