#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Data Clustering  Kaggle 
# 
# #### https://www.kaggle.com/vipulgandhi/kmeans-detailed-explanation/notebook?select=CC+GENERAL.csv

# In[22]:


#packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing as pp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[2]:


data = pd.read_csv('GENERAL.csv')
# View missing values (count)
data.isna().sum()


# In[3]:


# Overview
data.describe()


# In[9]:


# Correlation plot
sns.heatmap(data.corr(),
            xticklabels=data.columns,
            yticklabels=data.columns,
            linewidths=.5,
            cmap="YlGnBu"
           )


# In[10]:


# Pairplot - dispersion between variables
sns.pairplot(data)


# In[14]:


# Distribution of int64 variables
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.hist(data['CASH_ADVANCE_TRX'], 65, histtype='bar', stacked=True)
ax0.set_title('CASH_ADVANCE_TRX')

ax1.hist(data['PURCHASES_TRX'], 173, histtype='bar', stacked=True)
ax1.set_title('PURCHASES_TRX')

ax2.hist(data['TENURE'], 7, histtype='bar', stacked=True)
ax2.set_title('TENURE')

fig.tight_layout()
plt.show()


# ### Features generation (copy data)

# In[31]:


features = data.copy()
list(features)


# In[32]:


# Log-transformation

cols =  ['BALANCE',
         'PURCHASES',
         'ONEOFF_PURCHASES',
         'INSTALLMENTS_PURCHASES',
         'CASH_ADVANCE',
         'CASH_ADVANCE_TRX',
         'PURCHASES_TRX',
         'CREDIT_LIMIT',
         'PAYMENTS',
         'MINIMUM_PAYMENTS',
        ]

# Note: Adding 1 for each value to avoid inf values
features[cols] = np.log(1 + features[cols])

features.describe()


# In[33]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_principal)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[34]:


features.boxplot(rot=90, figsize=(30,10)) #check outliers


# In[35]:


#IRQ methodology

irq_score = {}

for c in cols:
    q1 = features[c].quantile(0.25)
    q3 = features[c].quantile(0.75)
    score = q3 - q1
    outliers = features[(features[c] < q1 - 1.5 * score) | (features[c] > q3 + 1.5 * score)][c]
    values = features[(features[c] >= q1 - 1.5 * score) | (features[c] <= q3 + 1.5 * score)][c]
    irq_score[c] = {
        "Q1": q1,
        "Q3": q3,
        "IRQ": score,
        "n_outliers": outliers.count(),
        "outliers_avg": outliers.mean(),
        "outliers_stdev": outliers.std(),
        "outliers_median": outliers.median(),
        "values_avg:": values.mean(),
        "values_stdev": values.std(),
        "values_median": values.median(),
    }
    
irq_score = pd.DataFrame.from_dict(irq_score, orient='index')

irq_score


# ### # Scale All features
# #### put all variables at the same scale, with mean zero and standard deviation equals to one

# In[43]:


# Remove CUST_ID (not usefull)
features.drop("CUST_ID", axis=1, inplace=True)


# In[47]:


for col in cols:
    features[col] = pp.scale(np.array(features[col]))

features.head()


# In[51]:


# Fill NAs by mean
features = features.fillna(features.mean())
features.isna().sum()


# ### Clustinr - K-means

# In[52]:


X = np.array(features)
Sum_of_squared_distances = []
K = range(1, 30)

for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[54]:


# Choose number of clusters

n_clusters = 8

clustering = KMeans(n_clusters=n_clusters,
                    random_state=0
                   )

cluster_labels = clustering.fit_predict(X)

# plot cluster sizes

plt.hist(cluster_labels, bins=range(n_clusters+1))
plt.title('# Customers -- Cluster')
plt.xlabel('Cluster')
plt.ylabel('Customers')
plt.show()

# Assing cluster number to features and original dataframe
features['cluster_index'] = cluster_labels
data['cluster_index'] = cluster_labels


# In[55]:


# Dispersion between clusterized data
# Pairplot - dispersion between variables
sns.pairplot(features, hue='cluster_index')


# In[ ]:




