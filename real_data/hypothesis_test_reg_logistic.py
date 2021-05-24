from load_data import *
from matplotlib import colors as mcolors
from random import seed
from functions import *
import time
from scipy.stats import chi2
import numpy as np
from LogisticRegression import LogisticRegression
import datetime
from scipy.stats import gamma
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def sigmoid_func(w, x_d):
    return 1 / (1 + np.exp(- w.T @ x_d))


if __name__ == "__main__":
    # LOAD DATASET
    ds = 'compas'
    ########

    replications = 1
    tau = 0.5

    data, data_label, data_sensitive = upload_data(ds=ds)

    # X_train, a_train, y_train, X_test, a_test, y_test = upload_data(ds=ds)
    
   
    
    emp_marginals = np.reshape(get_marginals(sensitives=data_sensitive, target=data_label), [2, 2])
    regs = np.linspace(0, 100, 40)
    ########
    SEED = 123456
    seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(2)
    print('SEED:' + str(SEED))
    thetas = []
    dists = np.zeros([replications, len(regs)])
    min_rej_thr = np.zeros([replications, len(regs)])
    score = np.zeros([replications, len(regs)])
    thetas_norm = np.zeros([replications, len(regs)])
    C_res = []
    print(data.shape)
    
    for rep in range(replications):
        
        X_train, a_train, y_train, X_test, a_test, y_test, threshold = stratified_sampling(X=data, a=data_sensitive,
                                                                                           y=data_label,
                                                                                           emp_marginals=emp_marginals,
                                                                                           n_train_samples=int(.7 * data.shape[0]))
        
        C_res.append(y_test)
        for cnt, reg in enumerate(regs):
            start = time.time()
            # clf = FairLogisticRegression(reg=reg, radius=0, fit_intercept=False)
            clf = LogisticRegression(reg=reg, fit_intercept=False)
            clf.fit(X=X_train, y=y_train)
            theta = clf.coef_
            w = clf.intercept_
            print("intercept:",w)
            w = np.array([w]).flatten()[0]
            tau = 1 / (1 + np.exp(w))
            # theta = np.zeros(theta.shape)
            # theta[theta.shape[0] - 1] = 1
            print(theta)
            
            data_tuple = [X_test, a_test, y_test]
            
            asympt = limiting_dist_EQOPP(data_tuple, theta, tau)
            thetas_norm[rep, cnt] = LA.norm(theta)
            min_rej_thr[rep, cnt] = asympt * chi2.ppf(.95, 1)
            thetas.append(theta)
            C_res.append(C_classifier(X_test, theta, tau))
            print(C_classifier(X_test, theta, tau).sum(), clf.predict(X_test).sum())
            

           
            dist  = calculate_distance_eqopp(data_tuple, theta, tau)

            # cnt2 += (dist > min_rej_thr[rep, cnt])

            dists[rep, cnt] = dist
            score[rep, cnt] = clf.score(X=X_test, y=y_test)
            end = time.time()
            time_remaining = (end - start) * (regs.shape[0] - cnt - 1) * (replications - rep)

            conversion = datetime.timedelta(seconds=time_remaining)
            print('Regularization====>' + str(reg) + ', Time remaining : ' + str(conversion))

    np.savetxt('results/results_LR' + 'dists_' + ds + str(replications) + '.out', dists ,fmt='%.2f')
    np.savetxt('results/results_LR' + 'min_rej_thr_' + ds + str(replications) + '.out', min_rej_thr ,fmt='%.2f')
    np.savetxt('results/results_LR' + 'coefs_' + ds + str(replications) + '.out', thetas ,fmt='%.3f')
    np.savetxt('results/results_LR' + 'coefs_norm_' + ds + str(replications) + '.out', thetas_norm ,fmt='%.3e')
    np.savetxt('results/results_LR' + 'regs_' + ds + str(replications) + '.out', regs ,fmt='%.2f')
    np.savetxt('results/results_LR' + 'scores_' + ds + str(replications) + '.out', score ,fmt='%.3f')
    np.savetxt('results/results_LR' + 'correlation_' + ds + str(replications) + '.out', np.corrcoef(C_res) ,fmt='%.2f')
    print(np.corrcoef(C_res))
    #print("cnt ",cnt2)


    #
    #
    # plt.switch_backend('agg')
    fontsize = 17
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.xlabel('$\lambda$',size=19)
    ax.tick_params(axis='x', labelsize=fontsize)

    lns1 = ax.plot(regs, np.array(min_rej_thr)[0] , '--', c=colors['deeppink'], lw=2, label='$\hat \eta_{0.95}$' ,alpha=.9 )
    lns2 = ax.plot(regs, np.array(dists)[0], c=colors['limegreen'],
                   label='$N \\times \mathcal{D}(\hat \mathbb{P}^N)$' ,
                   lw=2, alpha=.9)
    ax.tick_params(axis='y', labelcolor='k',labelsize=fontsize)
    # plt.yscale('log')
    ax.set_ylabel('Statistic value',fontsize=19)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    lns3 = ax2.plot(regs, score[0], colors['darkmagenta'], marker='o', markevery=2, label='Test\naccuracy', lw=2, alpha=.9)
    ax2.tick_params(axis='y', labelcolor=colors['darkmagenta'],labelsize=fontsize)
    ax2.set_ylabel('Accuracy',fontsize=19)
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.19, 1), shadow=True, ncol=1,fontsize = fontsize)
    from matplotlib import rcParams

    # xlabel = '$' + str(N) + ' \\times \mathcal{R}^{' + 'opp' + '} (\hat \mathbb{P}^{' + str(N) + '}, \hat{p}^{' + str(
    #         N) + '})$'

    # ax.ylim([0, 1])
    ax.grid('on', alpha=.5)
    #
    #ax.locator_params(axis='y', nbins=5)
    #ax.locator_params(axis='x', nbins=5)
    #rcParams.update({'font.size': 22})
    

    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times']
    
    plt.tight_layout()
    plt.savefig('figs/reg_'+ds+'.pdf')
    plt.show()
    


