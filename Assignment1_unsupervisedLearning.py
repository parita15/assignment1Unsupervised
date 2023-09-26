#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Question1


# In[2]:


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# In[3]:


#Load the mnist_784 dataset
mnist_dataset = fetch_openml('mnist_784', version=1, as_frame=False)


# In[4]:


#Display each digit from the mnist dataset
data = mnist_dataset.data.astype(np.uint8)
labels = mnist_dataset.target.astype(np.uint8)
plt.figure(figsize=(10, 10))
for i in range(10):
    digit_indices = np.where(labels == i)[0]
    for j in range(10):
        plt.subplot(10, 10, i * 10 + j + 1)
        plt.imshow(data[digit_indices[j]].reshape(28, 28), cmap='gray')
        plt.title(f"Digit: {i}")
        plt.axis('off')

plt.tight_layout()
plt.show()


# In[5]:


#Use PCA to retrieve the 1st and 2nd principal component
n_components = 2

pca_component = PCA(n_components=n_components)

pca_component.fit(data)

#Below will find the 1st and 2nd component from PCA
firstPrincipalComponent = pca_component.components_[0]
secondPrincipal_component = pca_component.components_[1]

#Below will find the explained variance ratio
explainedVarianceRatio = pca_component.explained_variance_ratio_

print(f"Explained Variance Ratio for 1st Principal Component: {explainedVarianceRatio[0]}")
print(f"Explained Variance Ratio for 2nd Principal Component: {explainedVarianceRatio[1]}")


# In[7]:


#Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane.
projectionComponent_1st = np.dot(data, firstPrincipalComponent)
projectionComponent_2nd = np.dot(data, secondPrincipal_component)

#This will plot the 1st principal component
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(projectionComponent_1st, np.zeros_like(projectionComponent_1st), c=labels, cmap='viridis', marker='o')
plt.title("Projection onto 1st Principal Component")
plt.xlabel("Projection Value")
plt.yticks([])

#This will plot the 2nd principal component
plt.subplot(1, 2, 2)
plt.scatter(projectionComponent_2nd, np.zeros_like(projectionComponent_2nd), c=labels, cmap='viridis', marker='o')
plt.title("Projection onto 2nd Principal Component")
plt.xlabel("Projection Value")
plt.yticks([])

plt.tight_layout()
plt.show()


# In[9]:


#This will reduce the dimensionality of the MNIST dataset down to 154 dimensions with the help of incremental PCA
n_components = 154

incremental_PCA = IncrementalPCA(n_components=n_components)

batch_size = 2000
for i in range(0, data.shape[0], batch_size):
    batch = data[i:i+batch_size]
    incremental_PCA.partial_fit(batch)

mnist_reduced = incremental_PCA.transform(data)


# In[11]:


#Display the original and compressed digits 
sample_indices = np.random.choice(mnist_reduced.shape[0], 10, replace=False)

plt.figure(figsize=(12, 5))

#This will plot the original data
plt.subplot(1, 2, 1)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 1)
    plt.imshow(incremental_PCA.inverse_transform(mnist_reduced[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')

#This will plot the compressed data
plt.subplot(1, 2, 2)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 11)
    plt.imshow(incremental_PCA.inverse_transform(mnist_reduced[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Compressed')
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[12]:


#Question2


# In[13]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[14]:


#The below will generate the Swiss roll dataset
noSamples = 1000
X, color = make_swiss_roll(noSamples, noise=0.2, random_state=42)


# In[15]:


#Plot the resulting generated Swiss roll dataset.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll Dataset")
plt.show()


# In[17]:


#Below is Kernel PCA (kPCA) is used with linear kernel 
kpcaLinear = KernelPCA(kernel="linear", n_components=2)

#Below is Kernel PCA (kPCA) is used with a RBF kernel
kpcaRbf = KernelPCA(kernel="rbf", gamma=0.04, n_components=2)

#Below is Kernel PCA (kPCA) is used with a sigmoid kernel.
kpcaSigmoid = KernelPCA(kernel="sigmoid", gamma=0.001, n_components=2)

XLinear = kpcaLinear.fit_transform(X)
XRbf = kpcaRbf.fit_transform(X)
XSigmoid = kpcaSigmoid.fit_transform(X)

plt.figure(figsize=(15, 5))

#The below will plot the linear kernel graph
plt.subplot(131)
plt.scatter(XLinear[:, 0], XLinear[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Linear Kernel")

#The below will plot the RBF kernel graph
plt.subplot(132)
plt.scatter(XRbf[:, 0], XRbf[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with RBF Kernel")

#The below will plot the Sigmoid kernel graph
plt.subplot(133)
plt.scatter(XSigmoid[:, 0], XSigmoid[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Sigmoid Kernel")

plt.tight_layout()
plt.show()


# In[18]:


#Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
#Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at 
#the end of the pipeline. Print out best parameters found by GridSearchCV.
n_samples = 1000
X, color = make_swiss_roll(n_samples, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, color, test_size=0.2, random_state=42)

threshold = np.median(color)  
y_train_binary = (y_train > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('kpca', KernelPCA(kernel='rbf')),  
    ('classifier', LogisticRegression(max_iter=1000))
])

param_grid = {
    'kpca__kernel': ['rbf', 'sigmoid', 'poly'],  
    'kpca__gamma': [0.001, 0.01, 0.1, 1.0, 10.0]  
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train_binary)

best_params = grid_search.best_params_
print("Best Parameters:")
print(best_params)

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)

#The below will print the accuracy score
accuracy = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy on Test Set: {accuracy:.2f}")


# In[19]:


#Plot the results from using GridSearchCV
scores = grid_search.cv_results_["mean_test_score"]
gammas = [params["kpca__gamma"] for params in grid_search.cv_results_["params"]]
plt.figure(figsize=(10, 6))
plt.scatter(gammas, scores, c=scores, cmap=plt.cm.viridis)
plt.colorbar(label="Mean Test Score")
plt.xlabel("Gamma")
plt.ylabel("Mean Test Score")
plt.title("GridSearchCV Results")
plt.show()


# In[ ]:




