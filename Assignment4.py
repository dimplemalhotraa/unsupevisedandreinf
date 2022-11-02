#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit


# In[2]:


olivetti = fetch_olivetti_faces()
data = olivetti.data
target = olivetti.target


# In[3]:


pca = PCA(n_components=0.99)


# In[4]:


data = pca.fit_transform(data)


# In[5]:


pca.n_components_


# In[6]:


gm1= GaussianMixture(n_components=260, n_init =10, random_state=4).fit(data)
labels1 = gm1.predict(data)


# In[7]:


gm2= GaussianMixture(n_components=260, covariance_type='tied', n_init =10, random_state=42).fit(data)
labels2 = gm2.predict(data)


# In[8]:


gm3 = GaussianMixture(n_components=260, covariance_type='diag',  n_init =10, random_state=42, reg_covar=(1e-3)).fit(data)
labels3 = gm3.predict(data)


# In[9]:


gm4 = GaussianMixture(n_components=260, covariance_type='spherical',  n_init =10, random_state=42).fit(data)
labels4 = gm3.predict(data)


# In[10]:


def plot_cov_ellipse(cov, pos, col='b', nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, color=col, alpha=0.2,**kwargs)
    ax.add_artist(ellip)
    return ellip


# In[11]:


estimators = dict((cov_type, GaussianMixture(n_components=3,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])
for (name,estimator) in estimators.items():
    estimator.fit(data[:,2:4])


# In[12]:


colors = ['red', 'blue', 'green']
markers = ['+','*','x']


# In[13]:


for (name,gmm) in estimators.items():
    for n, color in enumerate(colors):
        plt.scatter(data[:, 2], data[:, 3],marker=markers[n])
        if gmm.covariance_type == 'full':
            cov = gmm.covariances_[n]
        elif gmm.covariance_type == 'tied':
            cov = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            cov = np.diag(gmm.covariances_[n])
        elif gmm.covariance_type == 'spherical':
            cov = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]


# In[14]:


pos = estimator.means_[n]
plot_cov_ellipse(cov,pos,col=color)


# In[15]:


plt.title(name)
plt.show()


# In[16]:


print(gm1.bic(data))
print(gm1.aic(data))
print(gm1.score(data))


# In[17]:


print(gm2.score(data))
print(gm3.score(data))
print(gm4.score(data))


# In[18]:


gms_per_k = [GaussianMixture(n_components=k, n_init=20, random_state=42).fit(data)
             for k in range(1, 20)]


# In[19]:


bics = [model.bic(data) for model in gms_per_k]
aics = [model.aic(data) for model in gms_per_k]


# In[20]:


plt.figure(figsize=(8, 4))
plt.plot(range(1, 20), bics, "bo-", label="BIC")
plt.plot(range(1, 20), aics, "go--", label="AIC")
plt.xlabel("$k$")
plt.ylabel("Information Criterion")
plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
plt.annotate('Minimum',
             xy=(3, bics[2]),
             xytext=(0.35, 0.6),
             textcoords='figure fraction',
             fontsize=14,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.legend()
plt.show()


# In[21]:


min_bic = np.infty


# In[22]:


for k in range(1, 11):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=20,
                              covariance_type=covariance_type,
                              random_state=42).fit(data).bic(data)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type


# In[23]:


print(best_k)
print(best_covariance_type) 


# In[24]:


print("For instance 1: ")
print("Hard: ",gm1.predict(data))
print("Soft: ",gm1.predict_proba(data))    


# In[25]:


print("For instance 2: ")
print("Hard: ",gm2.predict(data))
print("Soft: ",gm2.predict_proba(data))    


# In[26]:


print("For instance 3: ")
print("Hard: ",gm3.predict(data))
print("Soft: ",gm3.predict_proba(data))    


# In[27]:


print("For instance 4: ")
print("Hard: ",gm4.predict(data))
print("Soft: ",gm4.predict_proba(data))    


# In[28]:


def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()


# In[29]:


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]


# In[30]:


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]


# In[31]:


print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)


# In[32]:


n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gm1.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)
plot_faces(gen_faces, y_gen_faces)


# In[33]:


n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]


# In[34]:


n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]


# In[35]:


n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
y_darkened = y_train[:n_darkened]


# In[36]:


X_bad_faces = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])


# In[37]:


plot_faces(X_bad_faces, y_bad)


# In[38]:


X_bad_faces_pca = pca.transform(X_bad_faces)


# In[39]:


gm3.score_samples(X_bad_faces_pca)


# In[40]:


gm3.score_samples(data[:10])


# In[41]:


pca.n_components_


# In[42]:


print("Convertion:", gm1.converged_)


# In[43]:


print("Iteration:", gm1.n_iter_)


# In[44]:


print(gm1.bic(data))
print(gm1.aic(data))


# In[45]:


print(gm1.score(data))
print(gm2.score(data))
print(gm3.score(data))
print(gm4.score(data))


# In[ ]:




