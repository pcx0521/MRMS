import pandas as pd
import mifs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import datasets
# load X and y

X = pd.read_excel(r'E:\paper\emp\array_xls_MAY\array.xls').values #, index_col=0
y = pd.read_excel(r'E:\paper\emp\array_xls_MAY\array.xls').values
y = y[:,38]
#XX = pd.read_excel(r'E:\paper\emp\xlsSet\ukf_roi_be_T_PAR_left.xls').values #, index_col=0
# X = XX[:,0:10]
# #y = pd.read_excel(r'E:\paper\adco\yyy_ukf_roi_be_T_PAR_left.xls').values
# y = XX[:,10]
# y=y.ravel()




# X = pd.read_excel(r'C:\Users\Administrator\Desktop\svm_data\47\mrmr_net_cor.xlsx').values #, index_col=0
# y = pd.read_excel(r'C:\Users\Administrator\Desktop\svm_data\my_y_vector.xlsx').values

#XX = pd.read_excel(r'E:\paper\emp\xlsSet\ukf_roi_be_T_PAR_left.xls').values #, index_col=0
# X = XX[:,0:10]
# #y = pd.read_excel(r'E:\paper\adco\yyy_ukf_roi_be_T_PAR_left.xls').values
# y = XX[:,10]
# y=y.ravel()



X= preprocessing.StandardScaler().fit(X).transform(X)

# define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR')#method='MRMR'

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector._support_mask

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X = feat_selector.transform(X)

#K Nearest Neighbor(KNN)
test=0
train=0
recall=0
precision=0
time=1000
AUC=0
f1_score=0
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut

for i in range(time):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # , random_state=4
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics

    Ks =range(1,19)
    cv_scores = []  # 用来放每个模型的结果值
    best_k=1
    best_score=0

    ConfustionMx = []
    for n in Ks:
        # Train Model and Predict
        neigh = KNeighborsClassifier(n_neighbors=n)
        cv = LeaveOneOut()
        loo=LeaveOneOut()
        scores = cross_val_score(neigh, X_train, y_train.ravel(), cv=loo,scoring='accuracy')  # 进行交叉验证
        loss = -cross_val_score(neigh, X_train, y_train.ravel(), cv=loo, scoring='neg_mean_squared_error')  # for regression    损失函数
        score = np.mean(scores)  # compute mean cross-validation accuracy
        mean_loss=np.mean(loss)

        if score > best_score:
            best_score = score
            best_k=n

    print("The best K:",best_k)

    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = best_k).fit(X_train,y_train.ravel())

    from sklearn.metrics import confusion_matrix, roc_curve

    yhat = neigh.predict(X_test)
    cm_test = confusion_matrix(yhat, y_test)

    y_hat_train = neigh.predict(X_train)
    cm_train = confusion_matrix(y_hat_train, y_train)

    per_train = metrics.accuracy_score(y_train, neigh.predict(X_train))
    per_test = metrics.accuracy_score(y_test, yhat)
    per_recall = metrics.recall_score(y_test, yhat)
    per_precision = metrics.precision_score(y_test, yhat)
    per_f1=2/((1/per_recall)+(1/per_precision))


    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
    print("Test Recall Accuracy: ", metrics.recall_score(y_test, yhat))
    print("Test Precision: ", metrics.precision_score(y_test, yhat))


    #print('Recall for test set for svm = {}'.format(cm_test[1][1] / (cm_test[1][0] + cm_test[1][1])))

    #print(X_train)
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    # Print Area Under the Curve (AUC).
    print("AUC:", auc(fpr, tpr), "\n")
    ######################################################################

    per_auc=auc(fpr, tpr)

    train = train + per_train
    test = test + per_test
    recall = recall + per_recall
    precision=precision+per_precision
    AUC=AUC+per_auc
    f1_score=f1_score+per_f1

train = train / time
test = test / time
recall = recall / time
precision=precision/time
AUC=AUC/time
f1_score=f1_score/time

print('last Accuracy for training  = {}'.format(train))
print('last Accuracy for test set = {}'.format(test))
print('last Recall for test set  = {}'.format(recall))
print('last Precision for test set  = {}'.format(precision))
print('f1_score  = {}'.format(f1_score))
print('roc_curve  = {}'.format(AUC))







