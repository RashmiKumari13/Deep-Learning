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
