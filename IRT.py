import numpy as np
import pandas as np
import matplotlib.pyplot as plt

from scipy.stats import bernoulli


def sigmoid(a):
    return 1/(1+np.exp(-a))

class Student(object):
    def __init__(self, theta):
        self.theta = theta

    def expected_response(self, item):
        return sigmoid(item.kcs@self.theta-item.difficulty)
    
    def response(self, item):
        return bernoulli.rvs(self.expected_response(item))
    
    def improvement(self,lrn_gains):
        self.theta += lrn_gains

class Item(object):
    def __init__(self, difficulty,kcs):
        self.difficulty = difficulty
        self.kcs = kcs
    def expected_response(self, student):
        return sigmoid(self.kcs@student.theta-self.difficulty)
    def response(self, student):
        return bernoulli.rvs(self.expected_response(student))        

    
def generate_student(theta_dim):
    return Student(np.random.randn(theta_dim))    

def generate_item(kcs_dim):
    kcs=np.random.dirichlet(np.ones(kcs_dim))
    difficulty = np.random.randn()
    return Item(difficulty,kcs)