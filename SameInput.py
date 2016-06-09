# -*-encoding: utf-8 -

import math
import numpy as np
import pandas as pd
import sklearn.metrics as met
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import statistics as stat
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    """
    We'll try to see if the order of the features matters in the backfitting method.
    We create a model with only one function f :
        y = f(x1) + f(x2)
    then we'll try the backfitting method
    """

    n_est=10

    #X then X2

    X = np.linspace(0, 10, 300).reshape((300,1)) + np.random.normal(size=(300, 1), loc=0, scale=1e-1)
    X2 = np.linspace(0, 10, 300).reshape((300,1)) + np.random.normal(size=(300, 1), loc=10, scale=1e-1)
    y = np.sin(X) + X2**2

    # plotting the true model y vs X1 and y vs X2

    plt.scatter(X, y, label='y vs X1')
    plt.legend()
    plt.show()
    plt.scatter(X2, y, label='y vs X2', c='red')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, X2, y)
    plt.show()

    rdf= RandomForestRegressor(n_estimators=n_est)
    rdf.fit(X, y.reshape((300,)))
    y_pred = rdf.predict(X)
    graphics = np.empty((300, 2))
    graphics[:300, 0] = X.reshape((300,))   # sorting the data for the plot Y_pred vs X1
    graphics.sort(axis=0)                   # otherwise it's a fuzzy line
    graphics[:300, 1] = y_pred.reshape((300,))[X.reshape((300,)).argsort(axis=0)]

    plt.plot(graphics[:, 0], graphics[:, 1], label='y_pred vs X1')
    plt.scatter(X, y, marker='x')
    plt.legend()
    plt.title('Number of trees used for the forest : n_estimator=%s'%n_est)
    plt.show()

    # then we fit a second regressor on the residuals

    residuals = y.reshape((300,)) - y_pred.reshape((300,))
    print(residuals.shape)
    rdf2=RandomForestRegressor(n_estimators=n_est)
    rdf2.fit(X2, residuals)
    y_pred2 = rdf2.predict(X2)

    graphics2 = np.empty((300, 2))
    graphics2[:300, 0] = X2.reshape((300,))  # sorting for the plot y_pred2 vs X2
    graphics2.sort(axis=0)
    graphics2[:300, 1] = y_pred2.reshape((300,))[X2.reshape((300,)).argsort(axis=0)]

    plt.scatter(X2, residuals, c='green', marker='x')
    plt.plot(graphics2[:, 0], graphics2[:, 1], label='y_pred2 vs X2', c='blue')
    plt.legend()
    plt.show()

    y_pred_total = y_pred.reshape((300,)) + y_pred2.reshape((300,))

    print('MSE=%f'%met.mean_squared_error(y, y_pred_total))





    # X2 then X

    rdf=RandomForestRegressor(n_estimators=n_est)
    rdf.fit(X2, y.reshape((300,))) # We start the backfitting with X2

    y_pred = rdf.predict(X2)
    graphics = np.empty((300, 2))
    graphics[:300, 0] = X2.reshape((300,))  # sorting the data for the plot Y_pred vs X1
    graphics.sort(axis=0)  # otherwise it's a fuzzy line
    graphics[:300, 1] = y_pred.reshape((300,))[X2.reshape((300,)).argsort(axis=0)]

    plt.plot(graphics[:, 0], graphics[:, 1], label='y_pred vs X2')
    plt.scatter(X2, y, marker='x')
    plt.legend()
    plt.title('Number of trees used for the forest : n_estimator=%s' % n_est)
    plt.show()

    residuals2 = y.reshape((300,)) - y_pred.reshape((300,))
    rdf2=RandomForestRegressor(n_estimators=n_est)
    rdf2.fit(X, residuals2)
    y_pred2 = rdf2.predict(X)
    graphics2 = np.empty((300, 2))
    graphics2[:300, 0] = X.reshape((300,))  # sorting for the plot y_pred2 vs X2
    graphics2.sort(axis=0)
    graphics2[:300, 1] = y_pred2.reshape((300,))[X.reshape((300,)).argsort(axis=0)]

    plt.scatter(X, residuals2, c='green', marker='x')
    plt.plot(graphics2[:, 0], graphics2[:, 1], label='y_pred2 vs X', c='blue')
    plt.legend()
    plt.show()


    plt.scatter(np.linspace(1, 300, 300), residuals2 - residuals , marker='x', c='green')
    plt.show()
    y_pred_total = y_pred.reshape((300,)) + y_pred2.reshape((300,))

    print('MSE_2=%f'%met.mean_squared_error(y, y_pred_total))




