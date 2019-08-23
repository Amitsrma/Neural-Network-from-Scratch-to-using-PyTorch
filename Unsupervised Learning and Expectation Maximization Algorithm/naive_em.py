"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def gaussian(x,meu,var,dimension):
        return 1/(2*np.pi*var)**(dimension/2)*np.exp(-1/(2*var)*\
                  np.linalg.norm(x-meu,axis=1)**2)

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    d=X.shape[1]

    n=X.shape[0]

    K = mixture.mu.shape[0]

    soft_counts=np.zeros((n,K))
    ll=np.zeros(K)
    for i in range(n):
        Likelihood=mixture.p*gaussian(X[i],mixture.mu,mixture.var,d)
        soft_counts[i]=Likelihood/np.sum(Likelihood)
        ll+=soft_counts[i]*np.log(np.sum(Likelihood))

    return soft_counts,np.sum(ll)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d=X.shape
    _,k=post.shape
    meu=np.zeros((k,d))
    variance=np.zeros((k,))
    n_hat=post.sum(axis=0)
    p=n_hat/len(X)
    for i in range((k)):
        meu[i]=1/n_hat[i]*np.sum(np.vstack(post[:,i])*(X),axis=0)
        variance[i]=1/(d*n_hat[i])*\
        np.sum(
                (post[:,i])*np.linalg.norm(X-meu[i],axis=1)**2,
                )
    return GaussianMixture(mu=meu,var=variance,p=p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    soft_counts, old_logLikelihood=estep(X=X,mixture=mixture)
    mixture_updated=mstep(X,soft_counts)
    count=0
    while 1:

        count+=1
        soft_counts,logLikelihood=estep(X,mixture_updated)
        mixture_updated=mstep(X,soft_counts)
        if (logLikelihood-old_logLikelihood) <= 10**(-6)*abs(logLikelihood):
            break
        old_logLikelihood=logLikelihood

    return mixture_updated,soft_counts,logLikelihood



X=np.array([[0.85794562 ,0.84725174],
 [0.6235637  ,0.38438171],
 [0.29753461 ,0.05671298],
 [0.27265629 ,0.47766512],
 [0.81216873 ,0.47997717],
 [0.3927848  ,0.83607876],
 [0.33739616 ,0.64817187],
 [0.36824154 ,0.95715516],
 [0.14035078 ,0.87008726],
 [0.47360805 ,0.80091075],
 [0.52047748 ,0.67887953],
 [0.72063265 ,0.58201979],
 [0.53737323 ,0.75861562],
 [0.10590761 ,0.47360042],
 [0.18633234 ,0.73691818]])

mu=np.array([[0.6235637  ,0.38438171],
 [0.3927848  ,0.83607876],
 [0.81216873 ,0.47997717],
 [0.14035078 ,0.87008726],
 [0.36824154 ,0.95715516],
 [0.10590761 ,0.47360042]])

var=np.array([0.10038354 ,0.07227467 ,0.13240693 ,0.12411825 
              ,0.10497521 ,0.12220856])

p=np.array([0.1680912  ,0.15835331 ,0.21384187 ,0.14223565 
            ,0.14295074 ,0.17452722])

#"""
post=np.array([[0.17354324 ,0.19408461 ,0.38136556 ,0.0569083  ,0.16250611 ,0.03159219],
 [0.39379907 ,0.08689908 ,0.32081103 ,0.04067548 ,0.04920547 ,0.10860986],
 [0.35788286 ,0.01907566 ,0.18709725 ,0.04472511 ,0.01732312 ,0.37389601],
 [0.19268431 ,0.18091751 ,0.11938917 ,0.12743323 ,0.09677628 ,0.28279951],
 [0.36304946 ,0.07311615 ,0.43750366 ,0.02729566 ,0.04877955 ,0.05025552],
 [0.07858663 ,0.37039817 ,0.08705556 ,0.14917384 ,0.21407078 ,0.10071502],
 [0.13662023 ,0.29150288 ,0.10750309 ,0.13944117 ,0.14926196 ,0.17567066],
 [0.04532867 ,0.37841271 ,0.06233585 ,0.17307275 ,0.2613835  ,0.07946652],
 [0.03479877 ,0.30116079 ,0.03560306 ,0.24675099 ,0.22083886 ,0.16084754],
 [0.1084787  ,0.35703165 ,0.12209296 ,0.12356811 ,0.19771701 ,0.09111156],
 [0.18151437 ,0.29042408 ,0.1775779  ,0.09728296 ,0.14845737 ,0.10474333],
 [0.30076285 ,0.15240546 ,0.34401968 ,0.04831719 ,0.08817504 ,0.06631978],
 [0.14424702 ,0.32662602 ,0.16265301 ,0.10373169 ,0.17686354 ,0.08587872],
 [0.12020157 ,0.14175102 ,0.06966009 ,0.17178204 ,0.09140514 ,0.40520014],
 [0.06707408 ,0.29382796 ,0.05528713 ,0.20393925 ,0.17797873 ,0.20189285]])

#LL:-5.592899
#"""
