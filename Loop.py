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

if __name__ == '__main__':
    #plt.ion()
    """
    We'll try to see if the order of the features matters in the backfitting method.
    We create a model with only one function f :
        y = f(x1) + f(x2)
    """
    theta_0 = 1e-1
    theta_L = 1e-3  # setting hyper-parameters for GP
    theta_U = 1
    nugget = 1e-13

    M = 500 # number of iteration in the backfitting
    n_est = 30
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=n_est,
                                           random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=n_est),
        "K-nn": KNeighborsRegressor(),
        "Ada Boost": AdaBoostRegressor(n_estimators=n_est, base_estimator=KNeighborsRegressor()),
        "SVR": svm.SVR(cache_size=500),
        "GP":  GaussianProcess(theta0=theta_0, thetaL=theta_L, thetaU=theta_U, nugget= nugget)

    }

    estimator = ESTIMATORS["SVR"]
    estimator2 = clone(ESTIMATORS["SVR"], safe= True)
    try:
        str = input("Which regression technique ? (Default is : SVR) : ")
        estimator = ESTIMATORS[str]
        estimator2= clone(estimator, safe=True)

    except:
        raise KeyError("Try those one those key : %s"%ESTIMATORS.keys())
    """"
    X = np.linspace(0, 10., 300).reshape((300, 1)) + np.random.normal(size=(300, 1), loc=0, scale=1e-2)
    X2 = np.linspace(0., 10., 300).reshape((300, 1)) + np.random.normal(size=(300, 1), loc=0, scale=1e-2)
    y = np.sin(np.sqrt(np.abs(X))) + np.sin(np.sqrt(np.abs(X2)))
    """

    X = np.linspace(0.5, 2., 300).reshape((300, 1)) + np.random.normal(size=(300, 1), loc=0, scale=5e-2)
    X2 = np.linspace(2., 300, 300).reshape((300, 1)) + np.random.normal(size=(300, 1), loc=0, scale=5e-2)
    y = 1/np.abs(X) + np.sin(X2)


    # plotting the true model
    plt.scatter(X, 1/np.abs(X), label='f_1(X1) vs X1' )
    plt.legend()
    plt.show()
    plt.plot(X2, np.sin(X2), label='f_2(X2) vs X2')
    plt.legend()
    plt.show()

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, X2, y)
    plt.show()
    """

    # X then X2
    y_pred2 = np.zeros(shape=(300,1)) # setting an array full of 0 for the first iteration
    mse=[]
    for m in np.arange(1, M):

        estimator.fit(X, y.reshape((300,)) - y_pred2.reshape((300,)))
        y_pred = estimator.predict(X)

        # then we fit a second regressor on the residuals
        residuals = y.reshape((300,)) - y_pred.reshape((300,))
        estimator2.fit(X2, residuals)
        y_pred2 = estimator2.predict(X2)
        y_pred_total=y_pred + y_pred2
        mse.append(met.mean_squared_error(y, y_pred_total))

    plt.plot(mse)
    plt.title("Evolution of MSE vs number of iterations")
    plt.show()
    graphics = np.empty((300, 2))
    graphics[:300, 0] = X.reshape((300,))  # sorting the data for the plot Y_pred vs X1
    graphics.sort(axis=0)  # otherwise it's a fuzzy line
    graphics[:300, 1] = y_pred.reshape((300,))[X.reshape((300,)).argsort(axis=0)]

    plt.plot(graphics[:, 0], graphics[:, 1], label='y_pred vs X1')
    plt.scatter(X, 1/np.abs(X), label='f_1(X1) vs X1' )
    plt.legend()
    plt.title('Number of trees used for the forest : n_estimator=%s' % n_est)
    plt.show()

    graphics2 = np.empty((300, 2))
    graphics2[:300, 0] = X2.reshape((300,))  # sorting for the plot y_pred2 vs X2
    graphics2.sort(axis=0)
    graphics2[:300, 1] = y_pred2.reshape((300,))[X2.reshape((300,)).argsort(axis=0)]

    plt.plot(graphics2[:, 0], graphics2[:, 1], label='y_pred2 vs X2', c='blue')
    plt.plot(X2, np.sin(X2), 'r--', label='f_2(X2) vs X2')
    plt.legend()
    plt.show()

    plt.scatter(np.arange(0, 300, 1), y)
    plt.scatter(np.arange(0, 300, 1), y_pred_total, marker='<', c='red')
    plt.show()

    # X2 then X
    y_pred2 = np.zeros(shape=(300, 1))  # setting an array full of 0 for the first iteration
    mse2=[]
    for m in np.arange(1, M):

        estimator.fit(X2, y.reshape((300,)) - y_pred2.reshape((300,))) # starting with feature 2
        y_pred = estimator.predict(X2)

        residuals2 = y.reshape((300,)) - y_pred.reshape((300,))
        estimator2.fit(X, residuals2)
        y_pred2 = estimator2.predict(X)
        y_pred_total=y_pred + y_pred2
        mse2.append(met.mean_squared_error(y, y_pred_total))

    graphics = np.empty((300, 2))
    graphics[:300, 0] = X2.reshape((300,))
    graphics.sort(axis=0)
    graphics[:300, 1] = y_pred.reshape((300,))[X2.reshape((300,)).argsort(axis=0)]

    plt.plot(mse2)
    plt.title("Evolution of MSE vs number of iterations")
    plt.show()

    plt.plot(graphics[:, 0], graphics[:, 1], label='y_pred vs X2')
    plt.plot(X2, np.sin(X2), 'r--', label='f_2(X2) vs X2')
    plt.legend()
    plt.title('Number of trees used for the forest : n_estimator=%s' % n_est)
    plt.show()

    graphics2 = np.empty((300, 2))
    graphics2[:300, 0] = X.reshape((300,))  # sorting for the plot y_pred2 vs X2
    graphics2.sort(axis=0)
    graphics2[:300, 1] = y_pred2.reshape((300,))[X.reshape((300,)).argsort(axis=0)]

    plt.plot(graphics2[:, 0], graphics2[:, 1], label='y_pred2 vs X', c='blue')
    plt.scatter(X, 1/np.abs(X))
    plt.legend()
    plt.show()

    y_pred_total = y_pred.reshape((300,)) + y_pred2.reshape((300,))
    plt.scatter(np.arange(0, 300, 1), y)
    plt.scatter(np.arange(0, 300, 1), y_pred_total, marker='<', c='red')
    plt.show()








