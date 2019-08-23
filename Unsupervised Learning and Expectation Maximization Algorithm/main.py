import numpy as np
import kmeans
import common
import naive_em
import em

import matplotlib.pyplot as plt
def plot_points(X,post,title):
    assignment=np.argmax(post,axis=1)
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=assignment)
    plt.grid()
    plt.title(title)
    plt.show()



X = np.loadtxt("toy_data.txt")
K=[1,2,3,4]
# TODO: Your code here
costs=[]
loglikelihoods=[]
bics=[]
for k in K:
    cost_seeds_=[]
    log_likelihood_=[]
    bic_=[]
    for seed in range(4):
        gauss_mixture,post=common.init(X=X,K=k,seed=seed)
        gauss_mixture_kmeans,post_kmeans,cost=kmeans.run(X=X,mixture=gauss_mixture,
                                               post=post)
        #print('for k =',k, "and seed=",seed, end=" ")
        #print("cost=",cost)
        gauss_mixture_em,post_em,loglikelihood=naive_em.run(X,
                                                            gauss_mixture,
                                                            post)
        bic_.append(common.bic(X,gauss_mixture_em,loglikelihood))
        log_likelihood_.append(loglikelihood)
        cost_seeds_.append(cost)
#        plot_points(X,post_kmeans,
#                    title="kmeans with k:"+str(k)+" seed:"+str(seed))
#        plot_points(X,post_em,
#                    title="em with k:"+str(k)+" seed:"+str(seed))
    bics.append(bic_)
    costs.append(cost_seeds_)
    loglikelihoods.append(log_likelihood_)
