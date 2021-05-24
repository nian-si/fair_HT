from load_data import *
# from matplotlib.colors import ListedColormap
# from matplotlib import colors as mcolors
# import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
# import os
from scipy.stats import chi2
from functions import *
import time
# import matplotlib.pyplot as plt
# import scipy.special as sps
import datetime
# import matplotlib
# from matplotlib import rcParams


def sigmoid_func(beta, x):
    return 1 / (1 + np.exp(- beta.T @ x))


if __name__ == "__main__":
    # LOAD DATASET
    ds = 'toy_correlated'
    # n_data = 1e6
    true_marginals = np.array(([.1, .1], [.4, .4])) ## np.array(([P_00, P_01], [P_10, P_11]))
    beta = np.array([0, 1])
    replications = 1000
    N_range = [100,1000,2000] #[10, 100, 1000, 5000]
    #########
    cnt = 0
    R = np.zeros([replications, len(N_range)])
    test_res = np.zeros([replications, len(N_range)])
    print('True marginals=' + str(true_marginals))
    print('Replications=' + str(replications))
    print('theta=' + str(beta))
    print('N_range=' + str(N_range))
    s_hat = [];
    
    for i, N in enumerate(N_range):
        SEED = 1122334455
        average_0 = 0
        average_1 = 0
        cnt = 0
        seed(SEED)
        np.random.seed(SEED)
        rng = np.random.RandomState(2)
        for rep in range(replications):
            start = time.time()
            marginals_rand = np.random.multinomial(N, true_marginals.flatten(), size=1) / N
            marginals_rand = np.reshape(marginals_rand, [2, 2])
            while not (marginals_rand[1, 1] * marginals_rand[0, 1]):
                marginals_rand = np.random.multinomial(N, true_marginals.flatten(), size=1) / N
                marginals_rand = np.reshape(marginals_rand, [2, 2])
            marginals_rand = np.reshape(marginals_rand, [2, 2])

            data, data_label, data_sensitive = upload_data(ds=ds, n_samples=N, marginals=marginals_rand)
            data_tuple = [data, data_sensitive, data_label]

            dist = calculate_distance_eqopp(data_tuple, np.array([0, 1]), 0.5)
            s_hat.append(dist)
            asympt = limiting_dist_EQOPP(data_tuple, np.array([0, 1]), 0.5)
            threshold = asympt * chi2.ppf(.9, 1)
            cnt = cnt + (dist>threshold)
            average_0 = average_0 + asympt / replications
            average_1 = average_1 + dist / replications
      
        print(cnt)
        print(average_0,average_1)
        '''
        plt.figure()
        counts__, bins__, _ = plt.hist(s_hat,
                             density=True,
                             bins = np.linspace(0,3,30),
                             range = [0,3],
                             alpha=0.2,
                             edgecolor='black',
                             linewidth=1.3,
                             stacked=True,
                             label='N=' + str(N),
                             cumulative=False)
        plt.show()
        '''
           
