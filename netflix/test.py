import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 12
n, d = X.shape
seed = 0

# TODO: Your code here
loglikelihoods=[]
#bics=[]
for k in [1,12]:
    log_likelihood_=[]
    for seed in range(5):

#        bic_=[]
        #for seed in range(4):
        gauss_mixture,post=common.init(X=X,K=k,seed=seed)
        #print('for k =',k, "and seed=",seed, end=" ")
        #print("cost=",cost)
        gauss_mixture_em,post_em,loglikelihood=em.run(X,
                                                      gauss_mixture,
                                                      post)
#            bic_.append(common.bic(X,gauss_mixture_em,loglikelihood))
        log_likelihood_.append(loglikelihood)

#    bics.append(bic_)

    loglikelihoods.append(log_likelihood_)
