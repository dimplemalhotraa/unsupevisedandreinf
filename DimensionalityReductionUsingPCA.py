#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


mnist = fetch_openml('mnist_784')
mnist.target = mnist.target.astype(np.uint8)


# In[3]:


x=mnist['data']
y=mnist['target']


# In[4]:


x.shape


# In[5]:


y.shape


# In[6]:


scaler=StandardScaler()
scaler.fit(x)
scaled_data=scaler.transform(x)


# In[13]:


pca=PCA(n_components=2)


# In[18]:


X_pca = pca.fit_transform(scaled_data)
X_pca.shape


# In[19]:


pca.fit(X_reduce)


# In[20]:


pca.explained_variance_ratio_


# In[22]:


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=mnist['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')


# In[23]:


n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(x, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(x)


# In[24]:


X_pca=inc_pca.inverse_transform(X_reduced)


# In[25]:


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    n_rows = (len(instances) - 1) // images_per_row + 1
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# In[27]:


plt.figure(figsize=(8,6))
plt.subplot(121)
plot_digits(x[::2100])
plt.subplot(122)
plot_digits(X_pca[::2100])
plt.tight_layout()


# In[23]:


X,t=make_swiss_roll(n_samples=1000, noise=0.2, random_state=28)


# In[24]:


axes = [-11.5, 14, -2, 23, -12, 15]
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$y_1$", fontsize=18)
ax.set_ylabel("$y_2$", fontsize=18)
ax.set_zlabel("$y_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()


# In[26]:


rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)


# In[27]:


X_reduced = rbf_pca.fit_transform(X)


# In[28]:


lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)


# In[29]:


X_reduced=lin_pca.fit_transform(X)
plt.subplot()
plt.title("Linear Kernel")
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.show()


# In[30]:


plt.figure(figsize=(6, 5))
X_reduced=rbf_pca.fit_transform(X)
plt.subplot()
plt.title("Rbf Kernel")
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)


# In[31]:


plt.figure(figsize=(6, 5))
X_reduced=sig_pca.fit_transform(X)
plt.subplot()
plt.title("Sigmoid Kernel")
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)


# In[32]:


from sklearn.model_selection import GridSearchCV
y=t>6.9


# In[33]:


clf = Pipeline([("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression(solver="lbfgs"))])


# In[34]:


param_grid = [{"kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]}]


# In[35]:


grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)


# In[36]:


print(grid_search.best_params_)


# In[37]:


grid_scores = grid_search.cv_results_


# In[38]:


rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.03, fit_inverse_transform=True)
plt.figure(figsize=(6, 5))


# In[39]:


X_reduced=rbf_pca.fit_transform(X)
plt.subplot()
plt.title("Rbf Kernel with best params")
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
rbf_pca


# In[40]:


results_df = pd.DataFrame(grid_scores)


# In[41]:


results_df = results_df.sort_values(by=["rank_test_score"])


# In[42]:


results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
results_df.info()


# In[43]:


model_scores = results_df.filter(regex=r"split\d*_test_score")


# In[44]:


fig, ax = plt.subplots()
sns.lineplot(
    data=model_scores.transpose().iloc[:5],
    dashes=False,
    palette="Set1",
    marker="o",
    alpha=0.5,
    ax=ax,
)
ax.set_xlabel("CV test fold", size=12, labelpad=10)
ax.set_ylabel("Model AUC", size=12)
ax.tick_params(bottom=True, labelbottom=True)
plt.show()


# In[45]:


print(f"Correlation of models:\n {model_scores.transpose().corr()}")


# In[46]:


X_reduced[850:855]


# In[ ]:




