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
        self.theta = theta  # student's  real ability
        self.learning_trace=[]  # learning trace of the student

        self.theta_dim=theta.shape[0]  #dimension we're in

#to estimate the expected correctnes probability of the student to an item
    def expected_response(self, item):
        return sigmoid(item.kcs@self.theta-item.difficulty)
    
    
    def expected_response_pluriel(self, items):
         return [sigmoid(self.expected_response(items[j])) for j in range(len(items))]
    
#to simulate several responses of the student to a list of items    
    def simulate_pluriel(self, items):

        b=self.expected_response_pluriel(items)
        a=(np.random.uniform(low=0, high=1, size=len(b))<b).astype(int)
        c=list(zip([items[j].id for j in range(len(items))],a))
        self.learning_trace+=c
        return a
#to simulate the response of the student to an item  and add it to the learning trace
    def response(self, item):
        a=bernoulli.rvs(self.expected_response(item))
        self.learning_trace.append((item.id,a))
        return a
#to improve the student theta    
    def improvement(self,lrn_gains):
        self.theta += lrn_gains

#to erase the last nb_events from the learning trace
    def erase_learning_trace(self,nb_events):
        self.learning_trace=self.learning_trace[:-nb_events] 

#to generate a random student
def generate_student(theta_dim):
    return Student(np.random.randn(theta_dim))    




## Child class for Beyasian Student
class BayesianStudent(Student):
    def __init__(self,theta, mu, sigma,Corpus):
        super().__init__(theta)
        self.sigma = sigma 
        self.mu= mu
        self.prior = multivariate_normal(mean=mu, cov=sigma) #prior distribution of theta estimate
        self.corpus=Corpus

#to saple a theta from the prior
    def sample_distribution(self):
        self.sampled_theta = self.prior.rvs()

#to estimate expected correctnes probability from a theta sampled from the prior
    def expected_response_bayesian(self, item):
        return sigmoid(item.kcs@self.sampled_theta-item.difficulty)
    
    def expected_response_pluriel_bayesian(self, items):
        
        return [sigmoid(self.expected_response_bayesian(items[j])) for j in range(len(items))]
    
#optimize for new prior
    def optimize_prior(self):

        def f(x):
            return -log_posterior_map(x,self.corpus,self.learning_trace,self.prior)
        w0 = self.prior.rvs()
        res = scipy.optimize.minimize(f, w0, method='BFGS')

        self.mu=res.x
        self.sigma=np.linalg.inv(res.hess_inv)
        #change prior
        self.prior = multivariate_normal(mean=self.mu, cov=self.sigma)

#to estimate correctnes probability with the proxi ( should be better than with the sampled theta)
    def expected_proxi(self,item):
        mu_a=self.mu@item.kcs-item.difficulty
        sigma_a= item.kcs@(self.sigma@item.kcs)
        return sigmoid(mu_a/np.sqrt(1+np.pi*sigma_a/8))   
    
    def expected_proxi_pluriel(self,items):
        return [self.expected_proxi(items[j]) for j in range(len(items))]
#calculate accuracy with the proxi    
    def accuracy_with_proxi(self,items):

        a=self.simulate_pluriel(items)
        b=self.expected_proxi_pluriel(items)
        return roc_auc_score(a,b)
#calculate accuracy with the sampling method
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


    
#optimization with maximum likelihood
    def optimize_irt(self):

        def f(x):
            return -log_posterior(x,self.corpus,self.learning_trace)
        w0 = np.random.randn(self.theta.shape[0])
        res = scipy.optimize.minimize(f, w0, method='BFGS')
        #change estimated theta
        self.estimated_theta=res.x

#to estimate the expected correctnes probability of the student to an item with estimated theta
    def expected_response_irt(self, item):
        return sigmoid(item.kcs@self.estimated_theta-item.difficulty)
    
    def expected_response_pluriel_irt(self, items):
            
            return [sigmoid(self.expected_response_irt(items[j])) for j in range(len(items))]
#to improve the estimated theta with a learning rate ( ELO update)    
    def estimated_irt_improve(self, item, response, lrn_rate):
    
        a=[sigmoid(item.kcs[j]*self.estimated_theta[j]-item.difficulty) for j in range(len(item.kcs))]
        error=response-a
        self.estimated_theta += lrn_rate*error
# to update estimated_theta iteratively for a list of items        
    def estimated_irt_improve_pluriel(self, items, lrn_rate):
        responses=self.simulate_pluriel(items)
        for j in range(len(items)):

            self.estimated_irt_improve(items[j],responses[j],lrn_rate)    

#to calculate accuracy with the estimated theta    
    def accuracy_with_irt_estimate(self,items):

        a=self.simulate_pluriel(items)
        b=self.expected_response_pluriel_irt(items)
        return roc_auc_score(a,b)



