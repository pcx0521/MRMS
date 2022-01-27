import pandas as pd
import mifs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc
# Random Forest Classifier
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the datasets
# import moxing as mox
import os
import sys
import warnings
from pandas.errors import EmptyDataError
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
import warnings
import pandas as pd
from pandas.errors import EmptyDataError
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from torch.autograd import Variable
from sklearn.metrics import recall_score,f1_score,accuracy_score
def load_data(subject_dir, csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    subjects = os.listdir(subject_dir)

    x = []
    y = []
    for subject in subjects:
        features_path = os.path.join(subject_dir, subject)
        if not os.path.exists(features_path) or not features_path.endswith('npy'):
            continue
        else:
            row = df.loc[subject.split('.')[0]]
            label = int(row['male'])

            x.append(np.load(features_path))
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    return x, y

def load_data1(subject_dir, csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    subjects = os.listdir(subject_dir)

    x = []
    y = []
    for subject in subjects:
        features_path = os.path.join(subject_dir, subject)
        if not os.path.exists(features_path) or not features_path.endswith('npy'):
            continue
        else:
            row = df.loc[subject.split('.')[0]]
            label = int(row['Label'])

            x.append(np.load(features_path))
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    return x, y

def load_data2(subject_dir, csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    subjects = os.listdir(subject_dir)

    x = []
    y = []
    for subject in subjects:
        features_path = os.path.join(subject_dir, subject)
        if not os.path.exists(features_path) or not features_path.endswith('npy'):
            continue
        else:
            row = df.loc[subject.split('.')[0]]
            label = int(row['remove'])

            x.append(np.load(features_path))
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    return x, y

class MyDataset(data.Dataset):
    def __init__(self, x, y, device):
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y)
        self.device = device

    def __getitem__(self, index):
        xi = self.x[index].to(self.device)
        yi = self.y[index].to(self.device)
        return xi, yi

    def __len__(self):
        return len(self.y)


train_x, gender = load_data(r'./train_data/train', r'./train_data/train_open.csv')
train_x, label = load_data1(r'./train_data/train', r'./train_data/train_open.csv')
# super_feature_male = np.load("./super_feature_male.npy")
# super_feature = np.load("./super_feature.npy")

# a, remove = load_data2(r'./train_data/train', r'./train_data/remove3.csv')


# feature = np.load("./superfeature_index.npy")
train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
# mean = np.mean(train_x, axis=0)
# std = np.std(train_x, axis=0)
# train_x = (train_x - mean) / std
# train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
# np.save('./model/mean.npy', mean)
# np.save('./model/std.npy', std)
# index = [i for i in range(len(train_x))]
# random.shuffle(index)
# train_x = train_x[index]
# train_y = label[index]

# new = train_x[:, ~np.all(train_x[1:] == train_x[:-1], axis=0)]
X = train_x[:, [13926, 13936, 13941, 13942, 13944, 13945, 13956, 13957, 13959, 13960, 13961, 13962, 13963, 13965, 13966, 13967, 13968, 13969, 13972, 13983, 13984, 13985, 13986, 13988, 13993, 13994, 13995, 13997, 13999, 14001, 14002, 14003, 14004, 14009, 14010, 14012, 14013, 14014, 14019, 14020, 14029, 14032, 14033, 14035, 14037, 14039, 14040, 14042, 14048, 14053, 14054, 14082, 14085, 14086, 14089, 14105, 14106, 14108, 14113, 14115, 14117, 14121, 14122, 14123, 14129, 14132, 14134, 14139, 14149, 14152, 14153, 14157, 14160, 14162, 14172, 14173, 14177, 14198, 14199, 14200, 14203, 14204, 14205, 14206, 14222, 14223, 14224, 14231, 14232, 14235, 14237, 14239, 14248, 14250, 14255, 14259, 14268, 14275, 14277, 14278, 14281, 14286, 14289, 14295, 14296, 14299, 14300, 14303, 14312, 14315, 14320, 14321, 14325, 14357, 14358, 14359, 14375, 14429, 14450, 14456, 14457, 14458, 14475, 14485, 14486, 14487, 14499, 14510, 14511, 14517, 14525, 14530, 14552, 14556, 14557, 14564, 14569, 14596, 14642, 14648, 14655, 14669, 14671, 14676, 14677, 14678, 14687, 14691, 14692, 14703, 14705, 14710, 14714, 14725, 14731, 14732, 14734, 14737, 14746, 14747, 14751, 14756, 14757, 14760, 14765, 14768, 14771, 14774, 14775, 14777, 14781, 14825, 14868, 14885, 14896, 14913, 14915, 14940, 14951, 14956, 14957, 14984, 15026, 15034, 15043, 15047, 15066, 15074, 15075, 15084, 15087, 15091, 15092, 15107, 15195, 15197, 15198, 15203, 15210, 15212, 15213, 15215, 15225, 15231, 15235, 15238, 15240, 15241, 15242, 15244, 15260, 15264, 15274, 15287, 15297, 15299, 15301, 15302, 15303, 15307, 15308, 15316, 15320, 15321, 15323, 15327, 15331, 15334, 15337, 15353, 15359, 15368, 15376, 15387, 15389, 15397, 15398, 15399, 15402, 15409, 15410, 15413, 15414, 15418, 15423, 15424, 15436, 15442, 15450, 15452, 15491, 15495, 15497, 15506, 15507, 15508, 15509, 15512, 15518, 15520, 15524, 15525, 15538, 15558, 15559, 15564, 15566, 15587, 15588, 15623, 15627, 15647, 15665, 15667, 15698, 15707, 15723, 15731, 15785, 15819, 15821, 15837, 15838, 15839, 15840, 15841, 15842, 15844, 15845, 15846, 15847, 15850, 15852, 15853, 15854, 15869, 15882, 15887, 15888, 15893, 15897, 15904, 15911, 15918, 15925, 15932, 15934, 15935, 15944, 15950, 15969, 15972, 15974, 15983, 15987, 15994, 15995, 16016, 16023, 16031, 16043, 16045, 16053, 16056, 16070, 16073, 16078, 16088, 16090, 16104, 16113, 16128, 16136, 16148, 16162, 16171, 16173, 16182, 16183, 16184, 16185, 16186, 16190, 16193, 16196, 16197, 16211, 16229, 16241, 16259, 16264, 16268, 16270, 16271, 16272, 16275, 16283, 16285, 16290, 16293, 16299, 16302, 16308, 16318, 16321, 16322, 16329, 16340, 16341, 16345, 16348, 16355, 16358, 16360, 16365, 16368, 16378, 16385, 16392, 16394, 16402, 16404, 16408, 16409, 16419, 16421, 16432, 16435, 16436, 16438, 16444, 16445, 16458, 16474, 16475, 16483, 16486, 16487, 16488, 16504, 16505, 16507, 16509, 16519, 16522, 16540, 16547, 16555, 16568, 16575, 16576, 16577, 16578, 16580, 16584, 16588, 16589, 16604, 16617, 16618, 16624, 16648, 16650, 16651, 16662, 16663, 16665, 16675, 16676, 16678, 16679, 16680, 16682, 16685, 16686, 16687, 16689, 16694, 16701, 16706, 16709, 16717, 16727, 16728, 16729, 16791, 16792, 16799, 16821, 16822, 16831, 16842, 16852, 16861, 16885, 16888, 16895, 16901, 16903, 16909, 16930, 16940, 16942, 16943, 16944, 16945, 16946, 16949, 16950, 16964, 16965, 16968, 16969, 16970, 16973, 16999, 17012, 17020, 17037, 17040, 17051, 17055, 17069, 17072, 17082, 17085, 17095, 17127, 17131, 17140, 17191, 17198, 17204, 17209, 17262, 17272, 17273, 17275, 17277, 17374, 17380, 17406, 17413, 17465, 17481, 17499, 17539, 17553, 17556, 17563, 17575, 17590, 17618, 17680, 17703, 17706, 17723, 17765, 17784, 17787, 17797, 17798, 17800, 17803, 17804, 17823, 17829, 17834, 17836, 17837, 17839, 17875, 17887, 17930, 17954, 17955, 17956, 17962, 17964, 17969, 27996, 27997, 27998, 28004, 28005, 28007, 28021, 28022, 28023, 28032, 28034, 28035, 28037, 28038, 28040, 28041, 28046, 28047, 28061, 28062, 28064, 28065, 28067, 28084, 28086, 28087, 28088, 28089, 28115, 28117, 28119, 28145, 28146, 28147, 28148, 28149, 28152, 28153, 28154, 28156, 28158, 28159, 28160, 28164]]
y = label
# load X and y

# X = pd.read_excel(r'C:\Users\Administrator\Desktop\svm_data\whole_X_table.xlsx').values #, index_col=0
# y = pd.read_excel(r'C:\Users\Administrator\Desktop\svm_data\my_y_vector.xlsx').values
#
X= preprocessing.StandardScaler().fit(X).transform(X)


# define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector('MRMR')#method='MRMR'

# find all relevant features
feat_selector.fit(X, y)

# # check selected features
feat_selector._support_mask
#
# # check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)

np.save("./super_feature_male.npy", X_filtered)
#K Nearest Neighbor(KNN)
# test=0
# train=0
# recall=0
# precision=0
# #AUC=0
# from sklearn.model_selection import train_test_split
# for i in range(10000):
#     X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)#, random_state=4
#     print ('Train set:', X_train.shape,  y_train.shape)
#     print ('Test set:', X_test.shape,  y_test.shape)
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics
#
# Ks = 20
# k_best=1;
# score_best=0;
# mean_acc = np.zeros((Ks - 1))
# std_acc = np.zeros((Ks - 1))
# ConfustionMx = [];
# for n in range(1, Ks):
#     # Train Model and Predict
#     neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train.ravel())
#     yhat = neigh.predict(X_test)
#     # print(X_test)
#     mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
#
#     std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
#     if mean_acc[n-1]>score_best:
#         k_best=n
#         score_best=mean_acc[n-1]
#
#     '''
#     plt.plot(range(1,Ks),mean_acc,'g')
#     plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
#     plt.legend(('Accuracy ', '+/- 3xstd'))
#     plt.ylabel('Accuracy ')
#     plt.xlabel('Number of Nabors (K)')
#     plt.tight_layout()
#     plt.show()
#     plt.close('all')  # 避免内存泄漏
#     '''
#
#     #Train Model and Predict
#     neigh = KNeighborsClassifier(n_neighbors = k_best).fit(X_train,y_train)
#
#     yhat = neigh.predict(X_test)
#
#     from sklearn.metrics import confusion_matrix, roc_curve
#
#     cm_test = confusion_matrix(yhat, y_test)
#
#     y_hat_train = neigh.predict(X_train)
#     cm_train = confusion_matrix(y_hat_train, y_train)
#
#     per_train = metrics.accuracy_score(y_train, neigh.predict(X_train))
#     per_test = metrics.accuracy_score(y_test, yhat)
#     per_recall = metrics.recall_score(y_test, yhat)
#     per_precision = metrics.precision_score(y_test, yhat)
#
#
#     print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
#     print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
#     print("Test Recall Accuracy: ", metrics.recall_score(y_test, yhat))
#     print("Test Precision: ", metrics.precision_score(y_test, yhat))
#
#
#     #print('Recall for test set for svm = {}'.format(cm_test[1][1] / (cm_test[1][0] + cm_test[1][1])))
#     print("The best K:",k_best)
#     #print(X_train)
#     fpr, tpr, thresholds = roc_curve(y_test, yhat)
#     # Print Area Under the Curve (AUC).
#     print("AUC:", auc(fpr, tpr), "\n")
#     ######################################################################
#
#     #per_auc=metrics.auc(fpr, tpr)
#
#     train = train + per_train
#     test = test + per_test
#     recall = recall + per_recall
#     precision=precision+per_precision
#
#     #AUC=AUC+per_auc
#
# train = train / 10000
# test = test / 10000
# recall = recall / 10000
# precision=precision/10000
#
# #AUC=AUC/10000
#
# print('last Accuracy for training  = {}'.format(train))
# print('last Accuracy for test set = {}'.format(test))
# print('last Recall for test set  = {}'.format(recall))
# print('last Precision for test set  = {}'.format(precision))

#print('roc_curve  = {}'.format(AUC))







