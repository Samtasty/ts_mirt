import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
from items import *


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def generate_learning_gains(theta_dim):
    lambda_ = 10.0  # Rate parameter
    return np.random.exponential(scale=1/lambda_, size=theta_dim)
#%%
## Parent class for Student
class Student(object):
    def __init__(self, theta):
        self.theta = theta  # student's  real ability
        self.learning_trace = []  # learning trace of the student

        self.theta_dim = theta.shape[0]  # dimension we're in
        self.rewards_list = []  # list of rewards
        self.regrets_list = []  # list of regrets
        self.expected_reward_list = []
  

    # to estimate the expected correctnes probability of the student to an item
    def expected_response(self, item):
        a=self.theta.copy()
        return sigmoid(item.kcs @a  - item.difficulty)

    def expected_response_pluriel(self, items):
        return [sigmoid(self.expected_response(items[j])) for j in range(len(items))]

    # to simulate several responses of the student to a list of items
    def simulate_pluriel(self, items):

        b = self.expected_response_pluriel(items)
        a = (np.random.uniform(low=0, high=1, size=len(b)) < b).astype(int)
        c = list(zip([items[j].id for j in range(len(items))], a))
        self.learning_trace += c
        return a

    # to simulate the response of the student to an item  and add it to the learning trace
    def response(self, item):
        a = bernoulli.rvs(self.expected_response(item))
        self.learning_trace.append((item.id, a))
        return a

    # to improve the student theta
    def improvement(self, lrn_gains):
        self.theta += lrn_gains

    # to erase the last nb_events from the learning trace
    def erase_learning_trace(self, cut_off):
        a=self.learning_trace.copy()

        self.learning_trace = a[cut_off:]

#%%    

# to generate a random student
def generate_student(theta_dim):
    return Student(np.random.randn(theta_dim))




