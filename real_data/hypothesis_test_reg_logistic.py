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
    emp_marginals = np.reshape(get_marginals(sensitives=data_sensitive, target=data_label), [2, 2])
    regs = np.linspace(0, 100, 50)
    ########
    SEED = 1122334455
    seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(2)
    print('SEED:' + str(SEED))
    dists = np.zeros([replications, len(regs)])
    min_rej_thr = np.zeros([replications, len(regs)])
    score = np.zeros([replications, len(regs)])
    thetas = np.zeros([replications, len(regs)])
    for rep in range(replications):
        X_train, a_train, y_train, X_test, a_test, y_test, threshold = stratified_sampling(X=data, a=data_sensitive,
                                                                                           y=data_label,
                                                                                           emp_marginals=emp_marginals,
                                                                                           n_train_samples=int(.7 * data.shape[0]))
        for cnt, reg in enumerate(regs):
            start = time.time()
            # clf = FairLogisticRegression(reg=reg, radius=0, fit_intercept=False)
            clf = LogisticRegression(reg=reg, fit_intercept=False)
            clf.fit(X=X_train, y=y_train)
            theta = clf.coef_
            data_tuple = [X_test, a_test, y_test]
            asympt = limiting_dist_EQOPP(data_tuple, theta, tau)
            thetas[rep, cnt] = LA.norm(theta)
            min_rej_thr[rep, cnt] = asympt * chi2.ppf(.95, 1)

           
            dist  = calculate_distance_eqopp(data_tuple, theta, tau)

            dists[rep, cnt] = dist
            score[rep, cnt] = clf.score(X=X_test, y=y_test)
            end = time.time()
            time_remaining = (end - start) * (regs.shape[0] - cnt - 1) * (replications - rep)

            conversion = datetime.timedelta(seconds=time_remaining)
            print('Regularization====>' + str(reg) + ', Time remaining : ' + str(
                    conversion))

    np.savetxt('results_LR' + 'dists_' + ds + str(replications) + '.out', dists)
    np.savetxt('results_LR' + 'min_rej_thr_' + ds + str(replications) + '.out', min_rej_thr)
    np.savetxt('results_LR' + 'coefs_' + ds + str(replications) + '.out', thetas)
    np.savetxt('results_LR' + 'regs_' + ds + str(replications) + '.out', regs)
    np.savetxt('results_LR' + 'scores' + ds + str(replications) + '.out', score)
    #
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('$\lambda$')

    # plt.xlabel('$\lambda$')

    lns1 = ax.plot(regs, np.array(min_rej_thr)[0] , '--', c=colors['deeppink'], lw=2, label='$\hat \eta_{0.95}$' ,alpha=.9 )
    lns2 = ax.plot(regs, np.array(dists)[0], c=colors['limegreen'],
                   label='$N \\times \mathcal{R}^{' + 'opp}' + '(\hat \mathbb{P}^N, \hat p^N)$' ,
                   lw=2, alpha=.9)
    ax.tick_params(axis='y', labelcolor='k')
    # plt.yscale('log')
    ax.set_ylabel('Statistic value')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    lns3 = ax2.plot(regs, score[0], colors['darkmagenta'], marker='o', markevery=2, label='Test accuracy', lw=2, alpha=.9)
    ax2.tick_params(axis='y', labelcolor=colors['darkmagenta'])
    ax2.set_ylabel('Accuracy')
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.12), shadow=True, ncol=3)
    from matplotlib import rcParams

    # xlabel = '$' + str(N) + ' \\times \mathcal{R}^{' + 'opp' + '} (\hat \mathbb{P}^{' + str(N) + '}, \hat{p}^{' + str(
    #         N) + '})$'

    # ax.ylim([0, 1])
    ax.grid('on', alpha=.5)
    #
    # ax.locator_params(axis='y', nbins=5)
    # ax.locator_params(axis='x', nbins=5)

    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times']
    rcParams.update({'font.size': 12})
    plt.tight_layout()
    plt.savefig('reg.pdf')
    plt.show()
    


