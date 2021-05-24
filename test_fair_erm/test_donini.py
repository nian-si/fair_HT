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
from linear_ferm import Linear_FERM
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP
from sklearn.model_selection import GridSearchCV
from collections import namedtuple
import sys


def sigmoid_func(w, x_d):
    return 1 / (1 + np.exp(- w.T @ x_d))
class DataSet:
    def __init__(self,X,y):
        self.target = y
        self.data = X
if __name__ == "__main__":
    # LOAD DATASET
    ds = 'drug'
    method = 'Welch'
    criterion = 'equalized_odds' # only equal opportunity for Welch's test
    print(ds)
    ########

    replications = 1000
    tau = 0.5

    data, data_label, data_sensitive = upload_data(ds=ds)
    # X_train, a_train, y_train, X_test, a_test, y_test  = upload_data(ds=ds) 


    cnt2 = 0
    
    emp_marginals = np.reshape(get_marginals(sensitives=data_sensitive, target=data_label), [2, 2])
    regs = np.linspace(0, 100, 2)
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
    cnt_pass_fair = 0
    cnt_pass_naive = 0
    
    for rep in range(replications):
        
        X_train, a_train, y_train, X_test, a_test, y_test, threshold = stratified_sampling(X=data, a=data_sensitive,
                                                                                           y=data_label,
                                                                                           emp_marginals=emp_marginals,
                                                                                           n_train_samples=int(.7 * data.shape[0]))
        
        C_res.append(y_test)
        
        for cnt, reg in enumerate(regs):
            if cnt == 0:
                print('\n naive: rep:',rep,"cnt fair",cnt_pass_fair,"cnt naive",cnt_pass_naive,"N:", X_train.shape[0])
            else:
                print('\n fair: rep:',rep,"cnt fair",cnt_pass_fair,"cnt naive",cnt_pass_naive,"N:", X_train.shape[0])
            start = time.time()
            # clf = FairLogisticRegression(reg=reg, radius=0, fit_intercept=False)
            param_grid = [{'C': [0.01, 0.1, 1.0], 'kernel': ['linear']}]
            svc = svm.SVC()
            clf = GridSearchCV(svc, param_grid, n_jobs=1)
            original_X_test = X_test
            

            if cnt == 0:   
                clf.fit(X_train, y_train)
                print('Best Estimator:', clf.best_estimator_) 
                print("theta:",clf.best_estimator_.coef_)
                w = clf.best_estimator_.intercept_
                print("intercept:",clf.best_estimator_.intercept_)
                pred = clf.predict(X_test)
                theta = clf.best_estimator_.coef_[0]
                score[rep, cnt] = clf.score(X=X_test, y=y_test)
            else:
                dataset = DataSet(X_train, y_train )
                algorithm = Linear_FERM(dataset, clf, a_train)
                algorithm.fit()
                print('Best Fair Estimator::', algorithm.model.best_estimator_)
                
                w = algorithm.model.best_estimator_.intercept_
                pred = algorithm.predict(X_test)
                
                
                theta = algorithm.model.best_estimator_.coef_[0]
                
                X_test = algorithm.new_representation(X_test)
               
                score[rep, cnt] = clf.score(X=X_test, y=y_test)
                
                print("theta:",theta)
                print("intercept:",w)
                print('DEO test:',equalized_odds_measure_TP(DataSet(original_X_test, y_test), algorithm, a_test, ylabel=1))
                


            
            tau = 1 / (1 + np.exp(w[0]))
            # print("tau:",tau)
            data_tuple = [X_test, a_test, y_test]
            
            # asympt = limiting_dist_EQOPP(data_tuple, theta, tau,phi_function, phi_function_derivative)
            
            thetas_norm[rep, cnt] = LA.norm(theta)
            # min_rej_thr[rep, cnt] = asympt * chi2.ppf(.95, 1)
            
            if method == 'Welch':
                min_rej_thr[rep, cnt] = chi2.ppf(.95, 1)
            if method == 'Ours':
                if criterion == 'equalized_odds':
                    asympt = limiting_dist_EQOPP_2d(data_tuple, theta, tau);
                    min_rej_thr[rep, cnt] = find_quantile(asympt[0],asympt[1],0.95);
                else:
                    asympt = limiting_dist_EQOPP(data_tuple, theta, tau,phi_function,phi_function_derivative,U_var);
                    min_rej_thr[rep, cnt] = asympt * chi2.ppf(.95, 1)


            thetas.append(theta)
            C_res.append(C_classifier(X_test, theta, tau))
            
            
            

            if method == 'Ours':
                if criterion == 'equalized_odds':
                    dist  = calculate_distance_eqopp_2d(data_tuple, theta, tau)  # equalized test
                else:
                    dist  = calculate_distance_eqopp(data_tuple, theta, tau)

            if method == 'Welch':
                dist = Welch_test(data_tuple, theta, tau)
            print("dist: ", dist, "threshold: ", min_rej_thr[rep, cnt])
            print("1d dist: ", calculate_distance_eqopp(data_tuple, theta, tau), "1d threshold: ", limiting_dist_EQOPP(data_tuple, theta, tau,phi_function,phi_function_derivative,U_var)* chi2.ppf(.95, 1))

            # cnt2 += (dist < min_rej_thr[rep, cnt])
            if dist < min_rej_thr[rep, cnt] or sum(pred) == 0 or sum(pred) == len(y_test):

                print(cnt," PASS THE TEST!!!!")
                if cnt == 0:
                    cnt_pass_naive += 1
                else:
                    cnt_pass_fair += 1

            dists[rep, cnt] = dist
            
            print('Accuracy test:', accuracy_score(y_test, pred),score[rep, cnt],1-sum(y_test)/len(y_test))
            print("average predict ones:",C_classifier(X_test, theta, tau).sum()/len(y_test))
            
            end = time.time()
            time_remaining = (end - start) * (regs.shape[0] - cnt - 1) * (replications - rep)

            #conversion = datetime.timedelta(seconds=time_remaining)
            #print('Regularization====>' + str(reg) + ', Time remaining : ' + str(conversion))
            X_test = original_X_test

    print("cnt ",cnt2)
    print("cnt fair",cnt_pass_fair,"cnt naive",cnt_pass_naive)
    print("acuracy naive:",np.mean(score[:, 0]),np.std(score[:, 0]))
    print("acuracy fair:",np.mean(score[:, 1]),np.std(score[:, 1]))
    print(ds,method, criterion)
    '''
    np.savetxt('results/results_LR' + 'dists_' + ds + str(replications) + '.out', dists ,fmt='%.2f')
    np.savetxt('results/results_LR' + 'min_rej_thr_' + ds + str(replications) + '.out', min_rej_thr ,fmt='%.2f')
    np.savetxt('results/results_LR' + 'coefs_' + ds + str(replications) + '.out', thetas ,fmt='%.3f')
    np.savetxt('results/results_LR' + 'coefs_norm_' + ds + str(replications) + '.out', thetas_norm ,fmt='%.3e')
    np.savetxt('results/results_LR' + 'regs_' + ds + str(replications) + '.out', regs ,fmt='%.2f')
    np.savetxt('results/results_LR' + 'scores_' + ds + str(replications) + '.out', score ,fmt='%.3f')
    np.savetxt('results/results_LR' + 'correlation_' + ds + str(replications) + '.out', np.corrcoef(C_res) ,fmt='%.2f')
    print(np.corrcoef(C_res))
    


    #
    #
    # plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('$\lambda$')

    lns1 = ax.plot(regs, np.array(min_rej_thr)[0] , '--', c=colors['deeppink'], lw=2, label='$\hat \eta_{0.95}$' ,alpha=.9 )
    lns2 = ax.plot(regs, np.array(dists)[0], c=colors['limegreen'],
                   label='$N \\times \mathcal{D}(\hat \mathbb{P}^N)$' ,
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
    plt.savefig('figs/reg.pdf')
    plt.show()
    '''
    


