from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

# Read the MC Dataset. See McBagOfWordsDataset.txt
# Train data
ftrain = open("McBagOfWordsDataset.Train.txt")
trainData = np.loadtxt(ftrain)
X_train = trainData[:, :1]
Y_train = trainData[:, 1]
ftrain.close()

# Test data
ftest = open("McBagOfWordsDataset.Test.txt")
testData = np.loadtxt(ftest)
X_test = testData[:, :1]
Y_test = testData[:, 1]
ftest.close()


# Y = []
# for _ in tY:
    # print _
    # #Y.append(int(_))
#print X
#print "==========="
#print Y

# Split the data into train and test (70% train and 30% test)
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print len(X_train), len(X_test), len(Y_train), len(Y_test)

# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # # setup marker generator and color map
    # markers = ('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # # plot the decision surface
    # x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           # np.arange(x2_min, x2_max, resolution))
    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Z = Z.reshape(xx1.shape)
    # plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    
    # for idx, cl in enumerate(np.unique(y)):
        # plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    # alpha=0.8, c=cmap(idx),
                    # marker=markers[idx], label=cl)
    
    
    # # highlight test samples
    # if test_idx:        
        # # else:
        # X_test, y_test = X[test_idx, :], y[test_idx]    
        # plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o',s=55, label='test set')
    # #plt.show()

# Perform scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state = 0)
lr.fit(X_train_std, Y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
Y_pred = lr.predict(X_test_std)
Y_predProb  = lr.predict_proba(X_test_std)

print "Total: ", len(Y_test)
print "Misclassified samples: %d" % (Y_test != Y_pred).sum()
correctAns = 0
for idx, pred in enumerate(Y_pred):
    if Y_test[idx] == Y_pred[idx] and Y_pred[idx] == 1:
        correctAns += 1
    
print "Correct answer samples: %d" % correctAns
print "Accuracy: ", accuracy_score(Y_test, Y_pred)

print Y_pred
print "==============="

y1Pred = []
for _ in Y_predProb:
    y1Pred.append(_[1])
    print _[1]

# plot_decision_regions(X_combined_std, Y_combined, classifier = lr, test_idx=range(105,150))
# plt.show()
