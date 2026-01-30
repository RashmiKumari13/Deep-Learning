from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
#now we generate data set
X,y=make_blobs(n_features=2,centers=2,n_samples=1000,random_state=12)
print(type(X[:,0]))
# we will visualize dataset
plt.figure(figsize=(6,6))
plt.scatter(X[:,0],X[:,1], c=y)
plt.title('GROUND TRUTH',fontsize=18)
plt.show()


#initializing weight
w=np.random.rand(3,1)
print(w)


#adding bias
X_bias=np.ones([X.shape[0], 3)
print(X_bias)
X_bias[:, 1:3]= X
print(X_bias)
