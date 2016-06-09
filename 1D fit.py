# -*-encoding: utf-8 -

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as met
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

if __name__ == '__main__':

    n = 300 # number of points

    X1 = np.linspace(0.5, 10., n).reshape((n, 1))
    X2 = np.random.uniform(200, 400, size=(n, 1))
    X2.sort(axis=0)
    #print("X2=%s"%X2)
    Y = X1**3 + np.log(X2) + np.exp((X1/(10*X2)))

    plt.scatter(np.arange(0, n, 1), Y)
    plt.show()

    plt.plot(X1, X1**3, label="X1**3 vs X1")
    plt.legend()
    plt.show()

    plt.plot(X2, np.log(X2), label="log(x2) vs X2")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1, X2, np.exp((X1/(10*X2))))
    plt.show()

    estimator = RandomForestRegressor(n_estimators=50)
    estimator.fit(X1, Y.reshape((n,)))
    Y_pred1 = estimator.predict(X1)
    Res_1 = Y.reshape((n,)) - Y_pred1.reshape((n,))

    plt.plot(X1, Y_pred1, label="1D fit : Y on X1")
    plt.scatter(np.arange(0, n, 1),Y)
    plt.legend()
    plt.show()

    plt.plot(Res_1, label='residuals from model 1')
    plt.legend()
    plt.show()

    estimator2 = clone(estimator)
    estimator2.fit(X2, Y.reshape((n,)))
    Y_pred2 = estimator2.predict(X2)
    Res_2 = Y.reshape((n,)) - Y_pred2.reshape((n,))

    plt.plot(X2, Y_pred2, label="1D fit Y on X2")
    plt.scatter(np.arange(0, n, 1),Y)
    plt.legend()
    plt.show()
    plt.plot(Res_2, label="residuals from model 2")
    plt.legend()
    plt.show()

    # Linear regression on top of the 2 regression
    reg = LinearRegression()
    #reg = RidgeCV()
    X = np.zeros(shape=(n, 2))
    X[:, 0] = Y_pred1.reshape((n,))
    X[:, 1] = Y_pred2.reshape((n,))
    reg.fit(X, Y.reshape((n,)))
    Y_pred_lin = reg.predict(X)

    plt.plot(Y_pred_lin, label='2-D fit Y on (M1, M2)', c='red')
    plt.scatter(np.arange(0, n, 1), Y, marker='x')
    plt.legend()
    plt.show()

    Res_lin = Y.reshape((n,)) - Y_pred_lin.reshape((n,))    # Getting the Residuals
    plt.plot(Res_lin, label='Residuals of the global regression')
    plt.legend()
    plt.show()
    print(reg.get_params(deep=True))



    """
    estimator3=clone(estimator)
    X=np.zeros(shape=(n, 2))
    X[:,0]=X1.reshape((n,))
    X[:,1]=X2.reshape((n,))
    estimator3.fit(X, Y.reshape((n,)))
    Y_pred = estimator3.predict(X)
    Res_3 = Y .reshape((n,)) - Y_pred.reshape((n,))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1, X2, Y_pred)
    plt.show()

    plt.plot(Res_3, label='Residuals from the 2D fit')
    plt.legend()
    plt.show()
    """

