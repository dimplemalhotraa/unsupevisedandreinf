#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[2]:


face, target = fetch_olivetti_faces(data_home='./', return_X_y=True)


# In[3]:


print(face[0, 0].dtype, target.dtype)


# In[4]:


print(face.shape, target.shape)


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(face, target, stratify=target, test_size=.2)
plt.subplot(1, 2, 1)
plt.hist(y_train)
plt.title('training set distribution')
plt.xlabel('class')
plt.ylabel('number')
plt.subplot(1, 2, 2)
plt.hist(y_test)
plt.title('test set distribution')
plt.xlabel('class')
plt.ylabel('number')
plt.show()


# In[15]:


model = LogisticRegression()


# In[16]:


param_grid = {'solver': ['lbfgs', 'saga']}


# In[17]:


grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
grid_search.fit(x_train, y_train)


# In[18]:


print('best estimator:')
print(grid_search.best_estimator_)


# In[19]:


print('best score:')
print(grid_search.best_score_)


# In[20]:


predict = grid_search.predict(x_test)
print(classification_report(y_test, predict))


# In[21]:


silhouette_avg_list = []
search_space = range(100, 201, 10)
for cluster in search_space:
    kmeans = KMeans(n_clusters=cluster)
    cluster_labels = kmeans.fit_predict(face)
    silhouette_avg = silhouette_score(face, cluster_labels)
    print('for n_clusters =', cluster, 'The average silhouette_score is:', silhouette_avg)
    silhouette_avg_list.append(silhouette_avg)
plt.plot(search_space, silhouette_avg_list)
plt.xlabel('cluster')
plt.ylabel('silhouette score')
plt.show()


# In[22]:


kmeans = KMeans(n_clusters=150)
face_reduced = kmeans.fit_transform(face)


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(face_reduced, target, stratify=target, test_size=.2)
plt.subplot(1, 2, 1)
plt.hist(y_train)
plt.title('training set distribution')
plt.xlabel('class')
plt.ylabel('number')
plt.subplot(1, 2, 2)
plt.hist(y_test)
plt.title('test set distribution')
plt.xlabel('class')
plt.ylabel('number')
plt.show()


# In[24]:


model = LogisticRegression(solver='saga')
param_grid = {'max_iter': [200, 400, 600, 1000]}


# In[25]:


grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
grid_search.fit(x_train, y_train)


# In[26]:


print('best estimator:')
print(grid_search.best_estimator_)


# In[27]:


print('best score:')
print(grid_search.best_score_)


# In[28]:


predict = grid_search.predict(x_test)
print(classification_report(y_test, predict))


# In[ ]:




