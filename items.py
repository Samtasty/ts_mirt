
import numpy as np
import pandas as pd




from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.optimize
from sklearn.metrics import roc_auc_score



def sigmoid(a):
    return 1/(1+np.exp(-a))

##class for an Item
class Item(object):
    def __init__(self, difficulty,kcs,id):
        self.difficulty = difficulty #item's difficulty
        self.kcs = kcs #kcs value
        self.id=id #item's id
#to estimate the expected correctnes probability of the student to an item        
    def expected_response(self, student):
        return sigmoid(self.kcs@student-self.difficulty)
    def response(self, student):
        return bernoulli.rvs(self.expected_response(student))        

    



#generate a random item and make sure that the kcs are a probability distribution

def generate_item(kcs_dim,id):
    kcs=np.random.dirichlet(np.ones(kcs_dim)) #kcs value
    difficulty = np.random.randn() #difficulty value
    return Item(difficulty,kcs,id)



## Class for a corpus of items
class Corpus(object):
    def __init__(self, nb_items,kcs_dim):
        self.nb_items = nb_items
        self.kcs_dim=kcs_dim

        self.dic_item={j:generate_item(kcs_dim,j) for j in range(nb_items)}
#add a random item to the corpus
    def add_random_item(self):
        self.dic_item[self.nb_items]=generate_item(self.kcs_dim,self.nb_items)
        self.nb_items+=1
#add an item to the corpus
    def add_item(self, item):
        self.dic_item[self.nb_items]=item
        self.nb_items+=1       
#delete an item in the corpus        
    def delete_item(self, item_id):
        del self.dic_item[item_id]
        self.nb_items-=1    
#choose a list of items from the corpus
    def list_items(self,list_index):
        return [self.dic_item[j] for j in list_index]
#get a signle item from the corpus
    def get_item(self,item_id):
        return self.dic_item[item_id]