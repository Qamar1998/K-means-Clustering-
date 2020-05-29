#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering Algorthim 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[2]:


n_samples = 1000
random_state = 2000
X, y = make_blobs(n_samples=n_samples, 
        random_state=random_state, 
        centers=3)

# Plot the random blob data
i =2002
plt.figure(figsize=(4, 4))
plt.scatter(X[:, 0], X[:, 1], s=4)
plt.title(f"No Clusters Assigned")


# # K-Means Algorithm
# 

# In[3]:


# Plot the data and color code based on clusters
for i in range(1,5):
    plt.figure(figsize=(4, 4))
    # Predicting the clusters
    km = KMeans(n_clusters=i, random_state=random_state).fit_predict(X)
# plotting the clusters
    plt.scatter(X[:, 0], X[:, 1], c=km, s=4)
    plt.title(f"Number of Clusters: {i}")
plt.show();

