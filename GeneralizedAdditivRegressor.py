from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from hyperopt import tpe, hp, STATUS_OK, space_eval
from hyperopt import fmin as hyperopt_fmin
import sklearn.metrics as met
from sklearn.base import clone

class GeneralizedAdditiveRegressor(object):
    """Fit Generalized Additive Model with symmetric backfitting (we actualize the residuals once per batch)
    This way, the order of the features doesn't matter.
    
    Parameters
    ---------
    
    smoothers : list of estimators fo the shape funtions. It could be any estimator with fit() and predict() functions implemented.
        Note that the length of this list (the number of estimators) has to be equal to the number of features ( one function per feature).
        If you want the same smoothers for all the shape functions, you can pass only the estimator (not in a list !) and it will create a         list of cloned estimator of length (n_features) before the fit. 
    
    max_iter : the number of iteration to run in the backfitting algorithm. (default is 10)
    
    ridge_alpha : the regularization coefficient for ridge regression upon the shape functions in the backfitting 
    (in order to rescale them)
    
    Attributes
    ----------
    
    smoothers_ : list of the fitted smoothers (the shape functions)
    
    ridge : ridge regressor used after each batch in the backfitting to rescale the shape functions
        Its parameter alphas is given by the user (ridge_alphas)
    
    """
    
    
    def __init__(self, smoothers, max_iter=10, ridge_alphas=10.):
        self.smoothers = smoothers
        self.max_iter = max_iter
        self.ridge_alphas = ridge_alphas
        
    def fit(self, X, y):
        """Fit the shape function of each features with the backfitting algorithm.
        Please note that the shape functions are centered (not reduced).
        
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples. 
            
        Returns
        -------
        self : object
            The Generalized Additive Model with the fitted shape functions
        """
        
        n_samples, n_features = X.shape
        
        if not isinstance(self.smoothers, list):
            self.smoothers_ = [clone(self.smoothers) for i in range(n_features) ]
            self.ridge = RidgeCV(alphas = [self.ridge_alphas]*len(self.smoothers_), fit_intercept=False)
        else:
            self.smoothers_ = [clone(self.smoothers[j]) for j in range(n_features) ]
            self.ridge = RidgeCV(alphas = [self.ridge_alphas]*len(self.smoothers_), fit_intercept=False)
            
        self.y_mean_ = np.mean(y)
        self.rmse_ = [] # array to stock the train error over the iteration
        y -= y.mean()
        temp = np.zeros(shape=(n_samples, n_features)) # array to stock the shape function for re-use in the next iteration
        shape_functions = np.zeros(shape=(n_samples, n_features))
        for i in range(self.max_iter):
            for j in range(n_features):
                # select all the columns except the j-th one
                idx = list(set(np.arange(0, n_features, 1)) - set([j])) 
                
                #Compute the residuals of the previous iteration          
                residuals = y.reshape((n_samples,1)) - temp[:, idx].sum(axis=1, keepdims=True).reshape((n_samples, 1)) 
                residuals -=residuals.mean()
                residuals = residuals
                #print(np.amin(residuals), np.amax(residuals), 'iteration number %s'%(i+1))
               
                self.smoothers_[j].fit(X[:, j:j+1], residuals.reshape((n_samples,))) #reshape cause deprecation warning
                shape_functions[:, j]= self.smoothers_[j].predict(X[:, j:j+1])
                shape_functions[:, j] -= shape_functions[:, j].mean()
            
            # RidgeRegression on top of the shape function in order to 're-scale' each shape functions
            self.ridge.fit(shape_functions, y)
            coef = self.ridge.coef_
            shape_functions *= coef
            
            y_pred = shape_functions.sum(axis=1)
            y_pred -= y_pred.mean()
            self.rmse_.append(met.mean_squared_error(y_pred, y))
            
            temp=shape_functions.copy()
            #plt.scatter(1, np.abs(residuals.min()), c='g', label='iteration = %s'%i)
            #plt.scatter(2, np.abs(residuals.max()), c='r')
            #plt.legend()
            #plt.show()
        return self

    
    def tranform(self, X):
        """ Transform function, return the prediction of the shape functions in each dimensions. 
        For pipeline use.
        
        Parameters
        ----------
        
        X : array like of shape (n_samples, n_features).
            data to be transformed
        
        Returns
        -------
        
        shape_functions : np.array of shape(n_ sampes, n_features).
            Prediction of each shape function for each feature.
        """
        
        n_samples, n_features = X.shape
        shape_functions = np.empty_like(X)
        for j in range(n_features):
            shape_functions[:, j] = self.smoothers_[j].predict(X[:, j])
        
        return shape_functions
        
        
    def predict(self, X):
        """ Predict regression target for X.
        
        The prediction is made using the GAM model (sum of the shape functions)
        """
        
        n_samples, n_features = X.shape
        y = np.ones(n_samples) * self.y_mean_
        for j in range(n_features):
            y += self.smoothers_[j].predict(X[:, j:j+1])
    
        return y
    



def obj_function(param):
    """obj_func for hyperopt on a GAM with 0.18 GP as smoothers
    return : global mse on the test set.
    """
    
    n_features = (len(param) - 2)//2 
    
    #print('param={0}'.format(param))
    shape_parameters = [0]*(2*n_features) #a list of dict to contain the parameters of each GP
    c=0
    i=0
    while i <= 2* n_features - 1 :
        #print('iteration nb : %s'%obj_function.nb_iter)
        shape_parameters[c] = {}
        shape_parameters[c]['constant_value'] = param[i]
        shape_parameters[c]['length_scale'] = param[i+1]
        i=i+2
        c+=1
    alpha_ = param[-2]  # all the GPs share the same nugget (noise variance estimation)
    ridge_alphas = param[-1]
    
    smoothers=[]
    for i in range(n_features):      
        constant_value = shape_parameters[i]['constant_value']
        length_scale = shape_parameters[i]['length_scale']
        
        smoothers.append(GaussianProcessRegressor(kernel=C(constant_value)*RBF(length_scale), alpha=alpha_, optimizer=None))
    
    gam = GeneralizedAdditiveRegressor(smoothers, max_iter=15, ridge_alphas=ridge_alphas)
    gam.fit(X_train, Y_train)
    y_pred = gam.predict(X_test)
    score =  met.mean_squared_error(Y_test - Y_test.mean(), y_pred - y_pred.mean())
    #print('score=%s \n'%score)
    return score


def create_space(n_feature):
    """ Function to create a search space for the hypers of the a sklearn 0.18 GP kernel.
    The order of the hypers is designed to work with  obj_function().
    
    Return a search space for hyperopt_fmin()"""
    
    n_sample = X_train.shape[0]
    size = [ X[:, i].max() - X[:, i].min() for i in range(n_feature)]
    
    constant_space = [hp.loguniform('constant_value%s'%i, np.log(10)*-2, np.log(10)*2) for i in range(n_feature)]
    l_scale_space = [hp.uniform('length_scale%s'%i, (size[i]/n_sample), size[i]) for i in range(n_feature)]
    space =[]
    #create an alternate search space for hyper-opt
    for j in range(n_feature):
        space.append(constant_space[j])
        space.append(l_scale_space[j])
    
    space.append(hp.loguniform('alpha', -11, 2))
    space.append(hp.loguniform('ridge', -2, 2))
    
    return space