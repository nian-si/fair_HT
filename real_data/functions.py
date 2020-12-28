import numpy.linalg as LA
import numpy as np
import multiprocessing as mp
import math
from functools import partial
import time
import math
gr = (math.sqrt(5) + 1) / 2
sigmoid_func = lambda beta, x: 1 / (1 + np.exp(- beta.T @ x))

def phi_function(data_sensitive, data_label, emp_marginals):
    return data_sensitive * data_label  /emp_marginals[1,1]- (1 - data_sensitive) * data_label/emp_marginals[0,1]

def phi_function_derivative(data_sensitive, data_label, emp_marginals):
    return [-data_sensitive * data_label  /(emp_marginals[1,1] ** 2), (1 - data_sensitive) * data_label/(emp_marginals[0,1]**2)]

def U_var(data_sensitive, data_label):
    return [data_sensitive * data_label,(1 - data_sensitive) * data_label]

def C_classifier(data, theta, tau):
    # logistics
    return (1./(1+np.exp(-np.dot(data, theta))) >= tau)

def Kernel(x,h):
    return 1/np.sqrt(2*math.pi) * np.exp (- (x/h)**2 / 2) / h

def distance_func(theta,x,w):
    return abs(np.dot(x, theta) - w)/np.linalg.norm(theta)

def limiting_dist_EQOPP(data_tuple, theta, tau):
    X = data_tuple[0]
    a = data_tuple[1]
    y = data_tuple[2]
    w = -math.log(1./tau - 1)
    marginals = np.reshape(get_marginals(sensitives=a, target=y), [2, 2])
    
    N = X.shape[0]
    # calculate var
    C_x = C_classifier(X, theta, tau)

    exp_first = np.mean(phi_function_derivative(a,y,marginals) * C_x, 1) # E[phi_z C]
    RV = exp_first @ U_var(a,y) + phi_function(a, y, marginals) * C_x 
    var = np.cov(RV)
    
    # calculate C
    h = N ** (-1/5.) 
    C = Kernel(distance_func(theta, X ,w) * (2 * C_x  - 1), h) * phi_function(a, y, marginals)
    C = np.dot(C, np.array(phi_function(a, y, marginals)).transpose()) / N
    
    return 1/2 / C * var # only work for one-dimensional phi


   





def get_marginals(sensitives, target):
    """Calculate marginal probabilities of test data"""
    N_test = sensitives.shape[0]
    P_11 = np.sum(
        [1 / N_test if sensitives[i] == 1 and target[i] == 1 else 0 for i
         in range(N_test)])
    P_01 = np.sum(
        [1 / N_test if sensitives[i] == 0 and target[i] == 1 else 0 for i
         in range(N_test)])
    P_10 = np.sum(
        [1 / N_test if sensitives[i] == 1 and target[i] == 0 else 0 for i
         in range(N_test)])
    P_00 = np.sum(
        [1 / N_test if sensitives[i] == 0 and target[i] == 0 else 0 for i
         in range(N_test)])
    if np.abs(P_01 + P_10 + P_11 + P_00 - 1) > 1e-10:
        print(np.abs(P_01 + P_10 + P_11 + P_00 - 1))
        print('Marginals are WRONG!')
    return P_00, P_01, P_10, P_11


def calculate_distance_eqopp(data_tuple, theta, tau):
    data = data_tuple[0]
    # print(data)
    data_sensitive = data_tuple[1]
    data_label = data_tuple[2]
    N = np.shape(data_sensitive)[0]
    emp_marginals = np.reshape(get_marginals(sensitives=data_sensitive, target=data_label), [2, 2])
    w = -math.log(1./tau - 1)
    d = distance_func(theta,data,w)
    C = C_classifier(data, theta, tau)
    
    phi = data_sensitive * data_label  /emp_marginals[1,1]- (1 - data_sensitive) * data_label/emp_marginals[0,1]
    
    s = - np.dot(C,phi).sum()
    # phi = phi_func((A==1 & Y==1),(A==0 & Y==1));
    # s = - (C.*phi).sum();
    # t = [sign(s) ./d .* (1 - 2* C) .* phi,d];
    
    t = np.array([np.sign(s) / d * (1 - 2 * C) * phi,d]).transpose()
    val = 0;
    t = t[np.argsort(-t[:,0])]
    # print(t)
    # print(s)
    s = np.sign(s) * s
    for i in range(N):
        if t[i,0] * t[i,1] <= s:
            s = s - t[i,0] * t[i,1]
            val = val + t[i,1]
        else:
            val = val + s/t[i,0]
            break
    return val

def stratified_sampling(X, a, y, emp_marginals, n_train_samples):
    emp_P_11 = emp_marginals[1, 1]
    emp_P_01 = emp_marginals[0, 1]
    emp_P_10 = emp_marginals[1, 0]
    emp_P_00 = emp_marginals[0, 0]
    X_11, X_01, X_10, X_00 = [], [], [], []
    for i in range(X.shape[0]):
        if a[i] == 1 and y[i] == 1:
            X_11.append(X[i, :])
        if a[i] == 0 and y[i] == 1:
            X_01.append(X[i, :])
        if a[i] == 1 and y[i] == 0:
            X_10.append(X[i, :])
        if a[i] == 0 and y[i] == 0:
            X_00.append(X[i, :])
    ind_11 = np.random.randint(low=0, high=np.array(X_11).shape[0], size=int(emp_P_11 * n_train_samples))
    ind_01 = np.random.randint(low=0, high=np.array(X_01).shape[0], size=int(emp_P_01 * n_train_samples))
    ind_10 = np.random.randint(low=0, high=np.array(X_10).shape[0], size=int(emp_P_10 * n_train_samples))
    ind_00 = np.random.randint(low=0, high=np.array(X_00).shape[0], size=int(emp_P_00 * n_train_samples))
    X_train_11 = np.array(X_11)[ind_11, :]
    X_train_01 = np.array(X_01)[ind_01, :]
    X_train_10 = np.array(X_10)[ind_10, :]
    X_train_00 = np.array(X_00)[ind_00, :]
    X_test11 = np.delete(np.array(X_11), ind_11, axis=0)
    X_test01 = np.delete(np.array(X_01), ind_01, axis=0)
    X_test10 = np.delete(np.array(X_10), ind_10, axis=0)
    X_test00 = np.delete(np.array(X_00), ind_00, axis=0)
    test_sensitives = np.hstack([[1] * X_test11.shape[0], [0] * X_test01.shape[0],
                                 [1] * X_test10.shape[0], [0] * X_test00.shape[0]])
    y_test = np.hstack([[1] * X_test11.shape[0], [1] * X_test01.shape[0],
                        [0] * X_test10.shape[0], [0] * X_test00.shape[0]])
    X_test = np.vstack([X_test11, X_test01, X_test10, X_test00])
    y_train = np.hstack([[1] * int(emp_P_11 * n_train_samples), [1] * int(emp_P_01 * n_train_samples),
                         [0] * int(emp_P_10 * n_train_samples), [0] * int(emp_P_00 * n_train_samples)])
    train_sensitives = np.hstack([[1] * int(emp_P_11 * n_train_samples), [0] * int(emp_P_01 * n_train_samples),
                                  [1] * int(emp_P_10 * n_train_samples), [0] * int(emp_P_00 * n_train_samples)])
    X_train = np.vstack([X_train_11, X_train_01, X_train_10, X_train_00])

    threshold = 1 - sum(y == 1) / y.shape[0]
    return X_train, train_sensitives, y_train, X_test, test_sensitives, y_test, threshold



    

    
