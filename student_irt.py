import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.optimize
from sklearn.metrics import roc_auc_score

from optimizations import log_posterior_map, log_posterior, laplace_approx
from items import *



def sigmoid(a):
    return 1/(1+np.exp(-a))



## Parent class for Student
class Student(object):
    def __init__(self, theta):
        self.theta = theta
        self.learning_trace=[]
        self.irt_theta=np.zeros(theta.shape)
        self.theta_dim=theta.shape[0]

    def expected_response(self, item):
        return sigmoid(item.kcs@self.theta-item.difficulty)
    
    # def expected_response_irt(self, item):
    #     return sigmoid(item.kcs@self.irt_theta-item.difficulty)

    
    def expected_response_pluriel(self, items):
        
        return [sigmoid(self.expected_response(items[j])) for j in range(len(items))]
    
    def simulate_pluriel(self, items):

        b=self.expected_response_pluriel(items)
        a=(np.random.uniform(low=0, high=1, size=len(b))<b).astype(int)
        c=list(zip([items[j].id for j in range(len(items))],a))
        self.learning_trace+=c
        return a
    
    def response(self, item):
        a=bernoulli.rvs(self.expected_response(item))
        self.learning_trace.append((item.id,a))
        return a
    
    def improvement(self,lrn_gains):
        self.theta += lrn_gains

    # def irt_improve(self, item, response, lrn_rate):
        
    #     a=[sigmoid(item.kcs[j]*self.irt_theta[j]-item.difficulty) for j in range(len(item.kcs))]
    #     error=response-a
    #     self.irt_theta += lrn_rate*error
        

    # def  irt_update(self, item, response, lrn_rate):
    #     b=self.expected_response(item)
    #     error=response-b
    #     self.improvement(lrn_rate*error)

    def erase_learning_trace(self,nb_events):
        self.learning_trace=self.learning_trace[:-nb_events] 


def generate_student(theta_dim):
    return Student(np.random.randn(theta_dim))    




## Child class for Beyasian Student
class BayesianStudent(Student):
    def __init__(self,theta, mu, sigma,Corpus):
        super().__init__(theta)
        self.sigma = sigma
        self.mu= mu
        self.prior = multivariate_normal(mean=mu, cov=sigma)
        self.corpus=Corpus

    def sample_distribution(self):
        self.sampled_theta = self.prior.rvs()


    def expected_response_bayesian(self, item):
        return sigmoid(item.kcs@self.sampled_theta-item.difficulty)
    
    def expected_response_pluriel_bayesian(self, items):
        
        return [sigmoid(self.expected_response_bayesian(items[j])) for j in range(len(items))]
    

    def optimize_prior(self):

        def f(x):
            return -log_posterior_map(x,self.corpus,self.learning_trace,self.prior)
        w0 = self.prior.rvs()
        res = scipy.optimize.minimize(f, w0, method='BFGS')

        self.mu=res.x
        self.sigma=np.linalg.inv(res.hess_inv)
        self.prior = multivariate_normal(mean=self.mu, cov=self.sigma)

    def expected_proxi(self,item):
        mu_a=self.mu@item.kcs-item.difficulty
        sigma_a= item.kcs@(self.sigma@item.kcs)
        return sigmoid(mu_a/np.sqrt(1+np.pi*sigma_a/8))   
    
    def expected_proxi_pluriel(self,items):
        return [self.expected_proxi(items[j]) for j in range(len(items))]
    
    def accuracy_with_proxi(self,items):

        a=self.simulate_pluriel(items)
        b=self.expected_proxi_pluriel(items)
        return roc_auc_score(a,b)

    def accuracy_with_sampling(self,items):
        a=self.simulate_pluriel(items)
        b=[]
        for j in range(len(items)):
            self.sample_distribution()
            b.append(self.expected_response_bayesian(items[j]))
    
        return roc_auc_score(a,b)
    




 ## Child class for Classical IRT Student
class Classical_IRT(Student):
    def __init__(self, theta,Corpus):
        super().__init__(theta)
        self.corpus=Corpus
        self.estimated_theta=np.zeros(theta.shape)

    

    def optimize_irt(self):

        def f(x):
            return -log_posterior(x,self.corpus,self.learning_trace)
        w0 = np.random.randn(self.theta.shape[0])
        res = scipy.optimize.minimize(f, w0, method='BFGS')
        self.estimated_theta=res.x

    def expected_response_irt(self, item):
        return sigmoid(item.kcs@self.estimated_theta-item.difficulty)
    
    def expected_response_pluriel_irt(self, items):
            
            return [sigmoid(self.expected_response_irt(items[j])) for j in range(len(items))]
    
    def estimated_irt_improve(self, item, response, lrn_rate):
    
        a=[sigmoid(item.kcs[j]*self.estimated_theta[j]-item.difficulty) for j in range(len(item.kcs))]
        error=response-a
        self.estimated_theta += lrn_rate*error
        
    def estimated_irt_improve_pluriel(self, items, lrn_rate):
        responses=self.simulate_pluriel(items)
        for j in range(len(items)):

            self.estimated_irt_improve(items[j],responses[j],lrn_rate)    
    
    def accuracy_with_proxi(self,items):

        a=self.simulate_pluriel(items)
        b=self.expected_response_pluriel_irt(items)
        return roc_auc_score(a,b)



