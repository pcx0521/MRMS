import pandas as pd
import mifs
import numpy as np

# load X and y

XX = pd.read_excel(r'E:\paper\emp\array_xls_MAY\array.xls').values #, index_col=0
#X = XX[1:49,0:38]
y = pd.read_excel(r'E:\paper\emp\array_xls_MAY\label.xlsx').values
#y = XX[1:49,38]
y=y.ravel()



# define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector('MRMR')#method='MRMR'

# find all relevant features
feat_selector.fit(XX, y)

# check selected features
feat_selector._support_mask

# check ranking of features
feat_selector.ranking_

#call transform() on X to filter it down to selected features
X = feat_selector.transform(XX)

data = pd.DataFrame(X)
writer = pd.ExcelWriter('X_filtered.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.6f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()

#svm
#importing the required modules

import warnings
from sklearn.model_selection import train_test_split, cross_val_score, \
    KFold, GridSearchCV, LeaveOneOut  # to split the data into train and test dataset
from sklearn.svm import SVC # model we are using for classification in this project
from sklearn.metrics import accuracy_score, roc_curve, auc  # to find the accuracy of our model
from sklearn.exceptions import ChangedBehaviorWarning
import numpy as np
import matplotlib.pyplot as plt

# Number of random trials
warnings.filterwarnings('ignore', category=ChangedBehaviorWarning)
#reading the data set

test=0
train=0
recall=0
precision=0

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3)#test_size = 0.2,,  random_state = 0
    '''

    from sklearn.preprocessing import StandardScaler as ss
    sc = ss()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    '''
#########################################   SVM   #############################################################
    best_score = 0
    for gamma in np.arange(0.01, 10, 0.1):
        for C in np.arange(0.01, 20, 0.1):
            svm = SVC(gamma=gamma, C=C,kernel='rbf',decision_function_shape='ovo')
            loo = LeaveOneOut()  #留一验证
            scores = cross_val_score(svm, X_train, y_train.ravel(),cv=loo)   #进行交叉验证
            score = np.mean(scores)   # compute mean cross-validation accuracy
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}


    from sklearn.svm import SVC
    classifier = SVC(**best_parameters)
    classifier.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
    print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
    print('Recall for test set for svm = {}'.format(cm_test[1][1]/(cm_test[1][0] + cm_test[1][1])),5)
    #############显示分类准确率图##############
    '''
    from sklearn import metrics
    from mlxtend.plotting import plot_confusion_matrix
    # Plot the confusion matrix.
    cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    plot_confusion_matrix(conf_mat=cm,
                        show_absolute=True,
                        show_normed=True,
                        colorbar=True,
                        figsize=(8,8))
    plt.show()
    #plt.close('all')  # 避免内存泄漏
    '''
######################################################
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    '''

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    '''
    # Print Area Under the Curve (AUC).
    print("AUC:", auc(fpr, tpr), "\n")
######################################################################
    per_train=(cm_train[0][0] + cm_train[1][1])/len(y_train)
    per_test=(cm_test[0][0] + cm_test[1][1])/len(y_test)
    per_recall=cm_test[1][1]/(cm_test[1][0] + cm_test[1][1])

    train=train+per_train
    test=test+per_test
    recall=recall+per_recall


train=train/10
test=test/10
recall=recall/10

print('last Accuracy for training set for svm = {}'.format(train))
print('last Accuracy for test set for svm = {}'.format(test))
print('last Recall for test set for svm = {}'.format(recall))







