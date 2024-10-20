import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
from scipy.stats import norm
import scipy.optimize
from sklearn.metrics import roc_auc_score


def sigmoid(a):
    return 1/(1+np.exp(-a))

class Student(object):
    def __init__(self, theta):
        self.theta = theta
        self.learning_trace=[]
        self.irt_theta=np.zeros(theta.shape)

    def expected_response(self, item):
        return sigmoid(item.kcs@self.theta-item.difficulty)
    
    def expected_response_irt(self, item):
        return sigmoid(item.kcs@self.irt_theta-item.difficulty)

    
    def expected_response_pluriel(self, items):
        
        return [sigmoid(self.expected_response(items[j])) for j in range(len(items))]
    
    def simulate_pluriel(self, items):

        b=self.expected_response_pluriel(items)
        a=(np.random.uniform(low=0, high=1, size=b.shape)<b).astype(int)
        c=list(zip([items[j].id for j in range(len(items))],a))
        self.learning_trace+=c
        return a
    
    def response(self, item):
        a=bernoulli.rvs(self.expected_response(item))
        self.learning_trace.append((item.id,a))
        return a
    
    def improvement(self,lrn_gains):
        self.theta += lrn_gains

    def irt_improve(self, item, response, lrn_rate):
        
        a=[sigmoid(item.kcs[j]*self.irt_theta[j]-item.difficulty) for j in range(len(item.kcs))]
        error=response-a
        self.irt_theta += lrn_rate*error
        

    def  irt_update(self, item, response, lrn_rate):
        b=self.expected_response(item)
        error=response-b
        self.improvement(lrn_rate*error)

    def erase_learning_trace(self,nb_events):
        self.learning_trace=self.learning_trace[:-nb_events] 

class Item(object):
    def __init__(self, difficulty,kcs,id):
        self.difficulty = difficulty
        self.kcs = kcs
        self.id=id
    def expected_response(self, student):
        return sigmoid(self.kcs@student-self.difficulty)
    def response(self, student):
        return bernoulli.rvs(self.expected_response(student))        

    
def generate_student(theta_dim):
    return Student(np.random.randn(theta_dim))    

def generate_item(kcs_dim,id):
    kcs=np.random.dirichlet(np.ones(kcs_dim))
    difficulty = np.random.randn()
    return Item(difficulty,kcs,id)




class BayesianStudent(Student):
    def __init__(self, mu, sigma,Corpus):
        super().__init__(mu)
        self.sigma = sigma
        self.mu= mu
        self.prior = norm(loc=mu, scale=sigma)
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
        self.sigma=-np.linalg.inv(solution.hess_inv)
        self.prior = norm(loc=self.mu, scale=self.sigma)

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


class Corpus(object):
    def __init__(self, nb_items,kcs_dim):
        self.nb_items = nb_items
        self.kcs_dim=kcs_dim

        self.dic_item={j:generate_item(kcs_dim,j) for j in range(nb_items)}

    def add_random_item(self):
        self.dic_item[self.nb_items]=generate_item(self.kcs_dim)
        self.nb_items+=1

    def add_item(self, item):
        self.dic_item[self.nb_items]=item
        self.nb_items+=1       
        
    def delete_item(self, item_id):
        del self.dic_item[item_id]
        self.nb_items-=1    
       

def log_posterior_map(w,corpus,learning_trace,prior):
    outcomes=[learning_trace[j][1] for j in range(len(learning_trace))]
    items_id=[learning_trace[j][0] for j in range(len(learning_trace))]   
    corrects=[np.log(corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    errors=[np.log(1-corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(learning_trace))])+prior.logpdf(w)
    return L

def log_posterior(w,corpus,learning_trace):
    outcomes=[learning_trace[j][1] for j in range(len(learning_trace))]
    items_id=[learning_trace[j][0] for j in range(len(learning_trace))]   
    corrects=[np.log(corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    errors=[np.log(1-corpus.dic_item[items_id[j]].expected_response(w)) for j in range(len(learning_trace))]
    L=np.sum([outcomes[j]*corrects[j]+(1-outcomes[j])*errors[j] for j in range(len(learning_trace))])
    return L

def laplace_approx(w, w_map, H):
    detH =  np.linalg.det(H)
    constant = np.sqrt(detH)/(2*np.pi)**(2.0/2.0)
    density = np.exp(-0.5 * (w-w_map).dot(H).dot(w-w_map))
    return constant * density