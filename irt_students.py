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



## Child class for Classical IRT Student
class Classical_IRT(Student):
    def __init__(self, theta, Corpus):
        super().__init__(theta)
        self.corpus = Corpus
        self.estimated_theta = np.zeros(theta.shape)
        self.theta_history = [np.zeros(theta.shape)]

    # optimization with maximum likelihood
    def optimize_irt(self):
        """optimize the IRT parameter follwing the maximum likelihood"""

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
        """returns correctness probability with the theta estimation

        Args:
            item (object): item to predict

        Returns:
            float: prediction
        """
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
        return max(rewards, key=rewards.get)

    # to improve the estimated theta with a learning rate ( ELO update)

    def get_best_reward(self):
        """returns the best expected reward to choose for the student with given theta

        Returns:
            int: reward of the best item
        """

        rewards = self.expected_reward_pluriel(
            self.corpus.list_items(list(range(self.corpus.nb_items)))
        )
        return max(rewards.values())

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
        b = {items[j].id:self.expected_reward(items[j].id) for j in range(len(items))}

        return b

    def estimated_irt_improve(self, item, response, lrn_rate):
        """elo update of IRT estimation after a response to an item

        Args:
            item (object): item given
            response (Boolean): has the student answered correctly or not
            lrn_rate (float): learning rate in the update
        """

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
        """Returns accuracy metrics with the IRT estimation

        Args:
            items (list): list of items to predict

        Returns:
            list: ROC_AUC, accuracy , MSE
        """

        a = self.simulate_pluriel(items)
        b = np.array(self.expected_response_pluriel_irt(items))
        d = np.array(self.expected_response_pluriel(items))
        c = (b > 0.5).astype(int)
        mean_acc = accuracy_score(a, c)
        mse = np.mean((d - b) ** 2)
        return roc_auc_score(a, b), mean_acc, mse
    
