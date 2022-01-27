import pandas as pd
import mifs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# load X and y

X = pd.read_excel(r'C:\Users\Administrator\Desktop\svm_data\new_network_X_table.xlsx').values #, index_col=0
y = pd.read_excel(r'C:\Users\Administrator\Desktop\svm_data\new_label.xlsx').values
#y=y.ravel()

from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)

'''
# define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector()#method='MRMR'

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector._support_mask

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X = feat_selector.transform(X)
'''
from matplotlib.colors import ListedColormap
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot()
ax.set_title("Input data")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
ax.set_xticks(())
ax.set_yticks(())
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid={"C":[0.1,0.8,0.99,1.01,5,10,100], "gamma":[0.01,0.1,0.8,0.95,1.01,10,100]}, cv=3)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))



