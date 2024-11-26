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

from student import *


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
        self.mu_history = [mu]
        self.sigma_history = [sigma]
        self.sampled_theta_history = [mu]
        self.mle_theta_history = []

    # to saple a theta from the prior
    def sample_distribution(self):
        self.sampled_theta = self.prior.rvs()
        self.sampled_theta_history.append(self.sampled_theta.copy())

    # to estimate expected correctnes probability from a theta sampled from the prior
    def expected_response_bayesian(self, item):
        return sigmoid(item.kcs @ self.sampled_theta - item.difficulty)

    def exploration_matrix(self, lambda_):
        """return the exploration matrix from the learning trace that allows to induce a particular geometry"""

        items_id, outcomes = zip(*self.learning_trace)
        H = lambda_ * np.eye(len(self.theta) + 1)
        for j in range(len(outcomes)):
            item = self.corpus.get_item(items_id[j])
            v = np.r_[item.kcs, -item.difficulty]
            outer_product = np.outer(v, v)

            H += outer_product
        inv_H = np.linalg.inv(H)
        self.V = H
        self.inv_V = inv_H

    def item_exploration_value(self, item_id):
        """return the exploration value of an item"""
        item = self.corpus.get_item(item_id)
        v = np.r_[item.kcs, -item.difficulty]
        return np.sqrt(v @ self.inv_V @ v)

    def fisher_info_item(self, item_id):
        item = self.corpus.get_item(item_id)
        v = np.r_[item.kcs, -item.difficulty]
        #similar to andrew lan
        outer_product = np.outer(v, v)

        denominateur = (
            2
            + np.exp(item.kcs @ self.mle_theta - item.difficulty)
            + np.exp(-item.kcs @ self.mle_theta + item.difficulty)
        )
        return outer_product / denominateur

    def expected_response_pluriel_bayesian(self, items):

        return [
            sigmoid(self.expected_response_bayesian(items[j]))
            for j in range(len(items))
        ]

    def expected_response_mle(self, item):
        """give the expected response to an item with the MLE theta

        Args:
            item (array): item to predict
        """
        return sigmoid(item.kcs @ self.mle_theta - item.difficulty)

    def expected_response_pluriel_mle(self, items):
        """give the expected response to a list of items with the MLE theta

        Args:
            items (list): list of items to predict
        """
        return [
            sigmoid(self.expected_response_mle(items[j])) for j in range(len(items))
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
        b = {items[j].id: self.expected_reward(items[j].id) for j in range(len(items))}

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

    def expected_reward_mle(self, item_id):
        """returns the expected reward of an item for the MLE theta of a student

        Args:
            item_id (int): id of the item

        Returns:
            float: value of the expected reward
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        return reward * self.expected_response_mle(item)

    def expected_reward_pluriel_mle(self, items):
        """returns the expected reward of a list of items for the MLE theta of a student

        Args:
            items (list): list of items id

        Returns:
            list: list of expected rewards
        """
        return {
            items[j].id: self.expected_reward_mle(items[j].id)
            for j in range(len(items))
        }

    def expected_reward_ucb(self, item_id, exploration_parameter):
        """returns the expected reward of an item for the UCB algorithm

        Args:
            item_id (int): id of the item
            exploration_function (function): function to calculate the exploration value

        Returns:
            float: value of the expected reward
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        return reward * self.expected_response(
            item
        ) + exploration_parameter * self.item_exploration_value(item_id)
    
    

    def expected_reward_pluriel_ucb(self, items, exploration_parameter):
        """returns the expected reward of a list of items for the UCB algorithm

        Args:
            items (list): list of items id
            exploration_function (function): function to calculate the exploration value

        Returns:
            list: list of expected rewards
        """
        return {
            items[j].id: self.expected_reward_ucb(items[j].id, exploration_parameter)
            for j in range(len(items))
        }

    def expected_reward_with_uncertainty(self, item_id,exploration_parameter):
        """returns the expected reward of an item for the UCB algorithm

        Args:
            item_id (int): id of the item
            exploration_function (function): function to calculate the exploration value

        Returns:
            float: value of the expected reward
        """
        item = self.corpus.get_item(item_id)
        reward = self.corpus.get_reward_item(item_id)
        return reward * self.expected_response(
            item
        ) + exploration_parameter * np.sqrt(self.fisher_info_item(item_id))
    
    def expected_reward_pluriel_with_uncertainty(self, items, exploration_parameter):
        """returns the expected reward of a list of items for the UCB algorithm

        Args:
            items (list): list of items id
            exploration_function (function): function to calculate the exploration value

        Returns:
            list: list of expected rewards
        """
        return {
            items[j].id: self.expected_reward_with_uncertainty(items[j].id, exploration_parameter)
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
        return max(rewards, key=rewards.get)

    def get_best_reward(self):
        """returns the best expected reward to choose for the student with given theta

        Returns:
            int: reward of the best item
        """

        rewards = self.expected_reward_pluriel(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards.values())

    def get_best_item_mle(self):
        """returns the best item to choose for the student with the MLE prior

        Returns:
            int: id of the best item
        """

        rewards = self.expected_reward_pluriel_mle(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards, key=rewards.get)

    def get_best_item_softmax(self, temperature):
        """returns the best item to choose for the student with the softmax algorithm

        Returns:
            int: id of the best item
        """

        rewards = self.expected_reward_plurie_mle(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        proba = {j: np.exp(rewards[j] / temperature) for j in rewards.keys()}
        proba = {j: proba[j] / sum(proba.values()) for j in proba.keys()}
        return np.random.choice(list(proba.keys()), p=list(proba.values()))

    def get_item_epsilon_greedy(self, epsilon):
        """returns the best item to choose for the student with the epsilon greedy algorithm

        Returns:
            int: id of the best item
        """
        if np.random.uniform() < epsilon:
            return np.random.choice(list(range(self.corpus.nb_items)))
        else:
            return self.get_best_item_mle()

    def get_best_item_ucb(self, exploration_parameter):
        """returns the best item to choose for the student with the UCB algorithm

        Returns:
            int: id of the best item
        """

        rewards = self.expected_reward_pluriel_ucb(
            self.corpus.list_items(list(range(self.corpus.nb_items))),
            exploration_parameter,
        )
        return max(rewards, key=rewards.get)

    def get_best_item_with_uncertainty(self, exploration_parameter):
        """returns the best item to choose for the student with the UCB algorithm

        Returns:
            int: id of the best item
        """

        rewards = self.expected_reward_pluriel_with_uncertainty(
            self.corpus.list_items(list(range(self.corpus.nb_items))),
            exploration_parameter,
        )
        return max(rewards, key=rewards.get)
    
    def simulate_bandit(self, item_id):
        """simulate the reward of the student to an item and add it to regret and reward list

        Args:
            item_id (int): id of the item

        Returns:
            float: reward of the student
        """
        item = self.corpus.get_item(item_id)
        response = self.response(item)
        reward = response * self.corpus.get_reward_item(item_id)

        expected_reward = self.expected_response(item).copy()
        best_action = self.get_best_reward().copy()
        self.expected_reward_list.append(expected_reward)

        self.rewards_list.append(reward)
        self.regrets_list.append(best_action - expected_reward)
        return reward

    # optimize for new prior
    def optimize_prior(self):
        """optimize the prior following the Lagrange Approximation"""

        def f(x):
            return -log_posterior_map(x, self.corpus, self.learning_trace, self.prior)

        w0 = self.prior.rvs()
        res = scipy.optimize.minimize(f, w0, method="BFGS")
        self.res = res

        self.mu = res.x
        self.sigma = np.linalg.inv(res.hess_inv)
        self.mu_history.append(self.mu.copy())
        self.sigma_history.append(self.sigma.copy())
        # change prior
        self.prior = multivariate_normal(mean=self.mu, cov=self.sigma)
        self.theta_history.append(self.prior)

    def classical_optimization(self):
        """optimize the prior following the Maximum Likelihood Estimation"""

        def f(x):
            return -log_posterior(x, self.corpus, self.learning_trace)

        w0 = np.zeros_like(self.theta)
        res = scipy.optimize.minimize(f, w0, method="BFGS")
        self.res = res
        self.mle_theta = res.x
        self.mle_theta_history.append(self.mle_theta.copy())

    # to estimate correctnes probability with the proxi ( should be better than with the sampled theta)
    def expected_proxi(self, item):
        """estimate the correctness probability of an item with the approximation

        Args:
            item (object): item to product

        Returns:
            float: probability of correctness
        """
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
        reward = self.corpus.get_reward_item(item_id)
        expected_proxi_outcome = self.expected_proxi(item).copy()
        return reward * expected_proxi_outcome

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
        return max(rewards, key=rewards.get)

    # calculate accuracy with the proxi
    def accuracy_with_proxi(self, items):
        """Returns accuracy metrics using the approximation

        Args:
            items (list): list of items to predict for the student

        Returns:
            tuple: ROC_AUC, mean accuracy, mean squared error
        """

        a = self.simulate_pluriel(items)
        b = np.array(self.expected_proxi_pluriel(items))
        d = np.array(self.expected_response_pluriel(items))
        c = (b > 0.5).astype(int)
        mean_acc = accuracy_score(a, c)
        mse = np.mean((d - b) ** 2)
        return roc_auc_score(a, b), mean_acc, mse

    # calculate accuracy with the sampling method
    def accuracy_with_sampling(self, items):
        """Returns accuracy metrics with the sampling method

        Args:
            items (list): list of items to predict for the student

        Returns:
            tuple: ROC_AUC, mean accuracy, mean squared error
        """
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

    def accuracy_with_mle(self, items):
        """Returns accuracy metrics with the MLE method

        Args:
            items (list): list of items to predict for the student

        Returns:
            tuple: ROC_AUC, mean accuracy, mean squared error
        """
        a = self.simulate_pluriel(items)
        d = np.array(self.expected_response_pluriel(items))
        b = np.array(self.expected_response_pluriel_mle(items))
        c = (b > 0.5).astype(int)
        mean_acc = accuracy_score(a, c)
        mse = np.mean((d - b) ** 2)
        return roc_auc_score(a, b), mean_acc, mse
