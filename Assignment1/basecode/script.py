import numpy as np
from numpy.core.fromnumeric import argmax, argmin
from numpy.lib.function_base import cov
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import csv


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    avg = [np.array([0, 0]) for i in range(6)]
    total = [np.array([0, 0]) for i in range(6)]
    for i in range(len(X)):
        total[int(y[i])][0] += 1
        total[int(y[i])][1] = total[int(y[i])][0]
        avg[int(y[i])] = avg[int(y[i])] + np.array(X[i])
    avg.pop(0)
    total.pop(0)
    means = np.transpose(np.divide(avg, total))
    # print(np.cov(np.transpose(X)))
    covmat = np.cov(np.transpose(X))
    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    # IMPLEMENT THIS METHOD
    avg = [np.array([0, 0]) for i in range(6)]
    total = [np.array([0, 0]) for i in range(6)]
    for i in range(len(X)):
        total[int(y[i])][0] += 1
        total[int(y[i])][1] = total[int(y[i])][0]
        avg[int(y[i])] = avg[int(y[i])] + np.array(X[i])
    avg.pop(0)
    total.pop(0)
    means = np.transpose(np.divide(avg, total))
    # print(np.cov(np.transpose(X)))
    output_classes = np.unique(y)
    covmats = []
    for output in output_classes:
        Xi = X[y.flatten() == output, :]
        covmat = np.cov(np.transpose(Xi))
        covmats.append(covmat)
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ypred_list = []
    mean = np.transpose(means)
    sig_inv = inv(covmat)
    for i in range(0, len(Xtest)):
        distances = []
        for j in range(0, len(mean)):
            distance = np.dot(
                np.dot(np.transpose(Xtest[i]-mean[j]), sig_inv), (Xtest[i]-mean[j]))
            distances.append(distance)
        ypred_list.append(float(argmin(distances)+1))

    ypred = np.asarray(ypred_list)
    ypred = ypred.reshape(len(ytest), 1)
    acc_count = 0
    for i in range(len(ytest)):
        if ypred[i] == ytest[i]:
            acc_count += 1
    acc = acc_count*100/len(ytest)

    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ypred_list = []
    mean = np.transpose(means)
    for i in range(0, len(Xtest)):
        distances = []
        for j in range(0, 5):
            distance = np.exp(-0.5*(np.dot(
                np.dot(np.transpose(Xtest[i]-mean[j]), inv(covmats[j])), (Xtest[i]-mean[j]))))/det(covmats[j])
            distances.append(distance)
        ypred_list.append(float(argmax(distances)+1))

    ypred = np.asarray(ypred_list)
    ypred = ypred.reshape(len(ytest), 1)
    acc_count = 0
    for i in range(0, len(ytest)):
        if ypred[i] == ytest[i]:
            acc_count += 1
    acc = acc_count*100/len(ytest)
    return acc, ypred


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    w = np.dot(inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    N = len(np.dot(np.transpose(X), X))
    I = np.identity(N)
    lambd_mat = np.dot(I, lambd)
    w = np.dot(inv((np.dot(np.transpose(X), X) + lambd_mat)),
               np.dot(np.transpose(X), y))
    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    sum = 0
    for i in range(len(Xtest)):
        sum += np.square(np.subtract(ytest[i],
                         np.dot(np.transpose(w), Xtest[i])))

    mse = sum/N
    return mse


def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    # print(w)
    w = w.reshape(65, 1)
    error = 0.5 * np.dot(np.transpose(y-np.dot(X, w)), (y-np.dot(X, w))) + \
        0.5 * np.dot(lambd, (np.dot(np.transpose(w), w)))
    error_grad = (np.dot(np.dot(np.transpose(X), X), w) -
                  np.dot(np.transpose(X), y) + lambd * w).flatten()

    return error, error_grad


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    # IMPLEMENT THIS METHOD
    Xp = np.ones((x.shape[0], p+1))
    i = 1
    while i < (p+1):
        Xp[:, i] = x ** i
        i = i+1
    return Xp

# Main script


# # Problem 1
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(
        open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0]*x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('QDA')


plt.show()
# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(
        open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)
mle_training = testOLERegression(w, X, y)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)
mle_i_training = testOLERegression(w_i, X_i, y)


print('MSE training data without intercept '+str(mle_training))
print('MSE testing data without intercept '+str(mle))
print('MSE training data with intercept '+str(mle_i_training))
print('MSE testing data with intercept '+str(mle_i))

# # Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
test_data = []
mses_data = []
lam = []
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    test_data.append([lambd, mses3_train[i], mses3[i]])
    mses_data.append(mses3[i])
    lam.append(lambd)
    i = i + 1
fields = ['LAMBDAS', 'TRAINING DATA', 'TESTING DATA']
filename = "Opt_Lambda.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(test_data)
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}    # Preferred value.
w_init = np.ones((X_i.shape[1], 1))

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True,
                   args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()


# Problem 5
pmax = 7
# lam[argmin(mses_data)] REPLACE THIS WITH lambda_opt estimated from Problem 3
lambda_opt = lam[argmin(mses_data)]
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()
