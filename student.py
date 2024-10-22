import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.optimize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from optimizations import log_posterior_map, log_posterior, laplace_approx
from items import *


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


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
        return sigmoid(item.kcs @ self.theta - item.difficulty)

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
    def erase_learning_trace(self, nb_events):
        self.learning_trace = self.learning_trace[:-nb_events]


# to generate a random student
def generate_student(theta_dim):
    return Student(np.random.randn(theta_dim))


## Child class for Beyasian Student
class BayesianStudent(Student):
    def __init__(self, theta, mu, sigma, Corpus):
        super().__init__(theta)
        self.sigma = sigma
        self.mu = mu
        self.prior = multivariate_normal(
            mean=mu, cov=sigma
        )  # prior distribution of theta estimate
        self.corpus = Corpus
        self.theta_history = [self.prior]

    # to saple a theta from the prior
    def sample_distribution(self):
        self.sampled_theta = self.prior.rvs()

    # to estimate expected correctnes probability from a theta sampled from the prior
    def expected_response_bayesian(self, item):
        return sigmoid(item.kcs @ self.sampled_theta - item.difficulty)

    def expected_response_pluriel_bayesian(self, items):

        return [
            sigmoid(self.expected_response_bayesian(items[j]))
            for j in range(len(items))
        ]

    def expected_reward(self, item_id):
        """returns the expected reward of an item for the real theta of a student

        Args:
            item_id (int): id of the item

        Returns:
            float: value of the expected reward
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        a = reward * self.expected_response(item)

        return a

    def expected_reward_pluriel(self, items):
        """returns the expected reward of a list of items for the real theta of a student

        Args:
            items (list): list of items id

        Returns:
            list: list of expected rewards
        """
        b = [self.expected_reward(items[j].id) for j in range(len(items))]

        return b

    def expected_reward_bayesian(self, item_id):
        """returns the expected reward of an item for the sampled theta of a student

        Args:
            item_id (int): id of the item

        Returns:
            float: value of the expected reward
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        return reward * self.expected_response_bayesian(item)

    def expected_reward_pluriel_bayesian(self, items):
        """returns the expected reward of a list of items for the sampled theta of a student

        Args:
            items (list): list of items id

        Returns:
            list: list of expected rewards
        """
        return {
            items[j].id: self.expected_reward_bayesian(items[j].id)
            for j in range(len(items))
        }

    def get_best_item_bayesian(self):
        """returns the best item to choose for the student with the bayesian prior

        Returns:
            int: id of the best item
        """

        rewards = self.expected_reward_pluriel_bayesian(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards)

    def get_best_reward(self):
        """returns the best expected reward to choose for the student with given theta

        Returns:
            int: reward of the best item
        """

        rewards = self.expected_reward_pluriel(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards)

    def simulate_bandit(self, item_id):
        """simulate the reward of the student to an item and add it to regret and reward list

        Args:
            item_id (int): id of the item

        Returns:
            float: reward of the student
        """
        item = self.corpus.get_item(item_id)

        reward = self.response(item) * self.corpus.get_reward_item(item_id)
        self.rewards_list.append(reward)
        self.regrets_list.append(self.get_best_reward().copy() - reward)
        return reward

    # optimize for new prior
    def optimize_prior(self):

        def f(x):
            return -log_posterior_map(x, self.corpus, self.learning_trace, self.prior)

        w0 = self.prior.rvs()
        res = scipy.optimize.minimize(f, w0, method="BFGS")
        self.res = res
        self.mu = res.x
        self.sigma = np.linalg.inv(res.hess_inv)
        # change prior
        self.prior = multivariate_normal(mean=self.mu, cov=self.sigma)
        self.theta_history.append(self.prior)

    # to estimate correctnes probability with the proxi ( should be better than with the sampled theta)
    def expected_proxi(self, item):
        mu_a = self.mu @ item.kcs - item.difficulty
        sigma_a = item.kcs @ (self.sigma @ item.kcs)
        return sigmoid(mu_a / np.sqrt(1 + np.pi * sigma_a / 8))

    def expected_proxi_pluriel(self, items):
        return [self.expected_proxi(items[j]) for j in range(len(items))]

    def expected_reward_proxi(self, item_id):
        """Returns expected reward with the proxi

        Args:
            item_id (int): id of item

        Returns:
            float: expected reward with proxi
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward(item_id)
        return reward * self.expected_proxi(item)

    def expected_reward_pluriel_proxi(self, items):
        """Returns expected reward with the proxi for a list of items

        Args:
            items (list): list of items

        Returns:
            dic: list of expected rewards
        """
        return {
            items[j].id: self.expected_reward_proxi(items[j].id)
            for j in range(len(items))
        }

    def get_best_item_proxi(self):
        """Returns the best item with the proxi

        Returns:
            int: id of the best item
        """
        rewards = self.expected_reward_pluriel_proxi(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards)

    # calculate accuracy with the proxi
    def accuracy_with_proxi(self, items):

        a = self.simulate_pluriel(items)
        b = np.array(self.expected_proxi_pluriel(items))
        d = np.array(self.expected_response_pluriel(items))
        c = (b > 0.5).astype(int)
        mean_acc = accuracy_score(a, c)
        mse = np.mean((d - b) ** 2)
        return roc_auc_score(a, b), mean_acc, mse

    # calculate accuracy with the sampling method
    def accuracy_with_sampling(self, items):
        a = self.simulate_pluriel(items)
        d = np.array(self.expected_response_pluriel(items))
        b = []
        for j in range(len(items)):
            self.sample_distribution()
            b.append(self.expected_response_bayesian(items[j]))
        b = np.array(b)
        c = (b > 0.5).astype(int)
        mean_acc = accuracy_score(a, c)
        mse = np.mean((d - b) ** 2)
        return roc_auc_score(a, b), mean_acc, mse


## Child class for Classical IRT Student
class Classical_IRT(Student):
    def __init__(self, theta, Corpus):
        super().__init__(theta)
        self.corpus = Corpus
        self.estimated_theta = np.zeros(theta.shape)
        self.theta_history = [np.zeros(theta.shape)]

    # optimization with maximum likelihood
    def optimize_irt(self):

        def f(x):
            return -log_posterior(x, self.corpus, self.learning_trace)

        w0 = np.random.randn(self.theta.shape[0])
        res = scipy.optimize.minimize(f, w0, method="BFGS")

        self.res = res
        # change estimated theta
        self.estimated_theta = res.x
        self.theta_history.append(np.copy(self.estimated_theta))

    # to estimate the expected correctnes probability of the student to an item with estimated theta
    def expected_response_irt(self, item):
        return sigmoid(item.kcs @ self.estimated_theta - item.difficulty)

    def expected_response_pluriel_irt(self, items):

        return [
            sigmoid(self.expected_response_irt(items[j])) for j in range(len(items))
        ]

    def expected_reward_irt(self, item_id):
        """Returns expected reward with IRT

        Args:
            item_id (int): id of item

        Returns:
            float: expected reward with IRT
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        return reward * self.expected_response_irt(item)

    def expected_reward_pluriel_irt(self, items):
        """Returns expected reward with IRT for a list of items

        Args:
            items (list): list of items

        Returns:
            dic: list of expected rewards
        """
        return {
            items[j].id: self.expected_reward_irt(items[j].id)
            for j in range(len(items))
        }

    def get_best_item_irt(self):
        """Returns the best item with IRT

        Returns:
            int: id of the best item
        """
        rewards = self.expected_reward_pluriel_irt(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards,key=rewards.get)

    # to improve the estimated theta with a learning rate ( ELO update)

    def get_best_reward(self):
        """returns the best expected reward to choose for the student with given theta

        Returns:
            int: reward of the best item
        """

        rewards = self.expected_reward_pluriel(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards)
    
    def expected_reward(self, item_id):
        """returns the expected reward of an item for the real theta of a student

        Args:
            item_id (int): id of the item

        Returns:
            float: value of the expected reward
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        a = reward * self.expected_response(item)

        return a
    
    def expected_reward_pluriel(self, items):
        """returns the expected reward of a list of items for the real theta of a student

        Args:
            items (list): list of items id

        Returns:
            list: list of expected rewards
        """
        b = [self.expected_reward(items[j].id) for j in range(len(items))]

        return b
    

    def estimated_irt_improve(self, item, response, lrn_rate):

        a = np.array(
            [
                sigmoid(item.kcs[j] * self.estimated_theta[j] - item.difficulty)
                for j in range(len(item.kcs))
            ]
        )
        v = a / sum(a)
        error = response - a
        # this allows to adjusts each coordinate of theta estimate separately
        self.rewards_list.append(response * self.corpus.get_reward_item(item.id))
        self.expected_reward_list.append(
            self.expected_response(item).copy() * self.corpus.get_reward_item(item.id)
        )
        self.regrets_list.append(
            self.get_best_reward()
            - self.expected_response(item) * self.corpus.get_reward_item(item.id)
        )
        self.estimated_theta += lrn_rate * error * v
        self.theta_history.append(np.copy(self.estimated_theta))

    # to update estimated_theta iteratively for a list of items
    def estimated_irt_improve_pluriel(self, items, lrn_rate):
        responses = self.simulate_pluriel(items)
        for j in range(len(items)):

            self.estimated_irt_improve(items[j], responses[j], lrn_rate)

    # to calculate accuracy with the estimated theta
    def accuracy_with_irt_estimate(self, items):

        a = self.simulate_pluriel(items)
        b = np.array(self.expected_response_pluriel_irt(items))
        d = np.array(self.expected_response_pluriel(items))
        c = (b > 0.5).astype(int)
        mean_acc = accuracy_score(a, c)
        mse = np.mean((d - b) ** 2)
        return roc_auc_score(a, b), mean_acc, mse
