import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np


import matplotlib.pyplot as plt
from student import Student

from dkt import *
from student import *

from optimizations import (
    log_posterior_map,
    log_posterior,
    reg_log_likelihood,
    ellipsoid_over_items,
)


class BayesianStudent(Student):
    def __init__(self, theta, mu, sigma, corpus, lambda_):
        super().__init__(theta)
        self.mu = mu
        self.sigma = sigma
        self.prior = multivariate_normal(mean=mu, cov=sigma)
        self.corpus = corpus
        self.lambda_ = lambda_
        self.usable_items_index = list(range(corpus.nb_items))

        # Histories
        self.mu_history = [mu]
        self.sigma_history = [sigma]
        self.prior_history = [self.prior]
        self.sampled_theta_history = [mu]
        self.mle_theta_history = []
        self.reg_mle_theta_history = []
        self.ellipsoid_theta_history = []
        self.exploration_hidden_state_matrix = {}
        self.inv_exploration_hidden_state_matrix = {}

        # Logs
        self.rewards_list = []
        self.regrets_list = []
        self.expected_reward_list = []

    ### Utility Methods ###

    def _get_item(self, item_id):
        """Fetch item object."""
        return self.corpus.get_item(item_id)

    def _get_reward(self, item_id):
        """Fetch reward value for the item."""
        return self.corpus.get_reward_item(item_id)

    ### Bayesian Sampling ###
    def sample_distribution(self):
        """Sample a new theta from the prior distribution."""
        sampled_theta = self.prior.rvs()
        self.sampled_theta_history.append(sampled_theta)
        return sampled_theta

    def get_trained_model(self, trained_model, item_hidden_states, item_outcomes):
        """put as attribute the trained model, the item hidden states and the item outcomes.

        Args:
            trained_model (_type_): dkt model used
            item_hidden_states (_type_): list of hidden states for each item
            item_outcomes (_type_): list of outcomes for each item
        """
        self.trained_model = trained_model
        self.item_hidden_states = item_hidden_states
        self.item_outcomes = item_outcomes
        for item_id in self.item_hidden_states.keys():

            self.exploration_hidden_state_matrix[item_id] = (
                exploration_from_hidden_state(
                    self.item_hidden_states[item_id], self.lambda_
                )
            )
            a = self.exploration_hidden_state_matrix[item_id].copy()
            self.inv_exploration_hidden_state_matrix[item_id] = np.linalg.inv(a)

    def get_fitted_models(self, classifier):
        self.classifier = classifier
        self.fitted_models = get_fitted_models(
            self.classifier, self.item_hidden_states, self.item_outcomes
        )

    def get_last_hidden_state(self):
        self.encoded_learning_trace, _ = preprocess_learning_trace(
            self.learning_trace, self.corpus.nb_items
        )
        _, (a, _) = self.trained_model.rnn(self.encoded_learning_trace)
        self.last_hidden_state = a.detach().numpy()

    ### Expected Responses ###

    def expected_response(self, item):
        """Expected correctness probability using real theta."""
        theta = self.theta.copy()
        return sigmoid(item.kcs @ theta - item.difficulty)

    def expected_response_bayesian(self, item):
        """Expected correctness probability using sampled theta."""
        sampled_theta = self.sampled_theta_history[-1].copy()
        return sigmoid(item.kcs @ sampled_theta - item.difficulty)

    def expected_response_mle(self, item):
        """Expected correctness probability using MLE theta."""
        mle_theta = self.mle_theta.copy()
        return sigmoid(item.kcs @ mle_theta - item.difficulty)

    def expected_response_proxi(self, item):
        """Expected correctness probability using approximation."""
        item_vector = item.get_vector()
        mu = self.mu.copy()
        sigma = self.sigma.copy()
        mu_a = mu @ item_vector[:-1] + item_vector[-1]
        sigma_a = item_vector[:-1] @ (sigma @ item_vector[:-1])
        return sigmoid(mu_a / np.sqrt(1 + np.pi * sigma_a / 8))

    def expected_response_ellipsoid(self, item):
        """expected resposne given the theta from ellipsoid minimization"""
        ellipsoid_theta = self.ellipsoid_theta.copy()
        return sigmoid(item.kcs @ ellipsoid_theta - item.difficulty)

    def expected_response_hidden_state(self, item):
        """expected response using hidden state from dkt"""
        last_hidden_state = self.last_hidden_state.copy()
        return (
            self.fitted_models[item.id].predict_proba(last_hidden_state.reshape(1, -1))[
                0
            ][1],
            last_hidden_state,
        )

    ### Exploration Matrix ###
    def exploration_matrix(self):
        """Compute the exploration matrix."""
        items_id, _ = zip(*self.learning_trace)
        items = [self._get_item(item_id) for item_id in items_id]
        v_matrix = np.array([item.get_vector()[:-1] for item in items])

        H = self.lambda_ * np.eye(len(self.theta)) + v_matrix.T @ v_matrix
        self.V = H
        self.inv_V = np.linalg.inv(H)  # Cached for reuse

    def update_exploration_matrix(self, item_id):
        """Update the exploration matrix with a new item."""
        item = self._get_item(item_id)
        v = item.get_vector()[:-1]
        self.inv_V = self.inv_V - np.outer(self.inv_V @ v, v @ self.inv_V) / (
            1 + v @ self.inv_V @ v
        )

    def item_exploration_value(self, item_id):
        """Compute the exploration value of an item."""
        item = self._get_item(item_id)
        v = item.get_vector()[:-1]
        return np.sqrt(v @ self.inv_V @ v)

    def fisher_information_matrix(self):
        """Compute the Fisher information matrix."""
        items_id, _ = zip(*self.learning_trace)
        items = [self._get_item(item_id) for item_id in items_id]
        fisher_matrix = np.zeros((len(self.theta), len(self.theta)))
        mle_theta=self.mle_theta.copy()
        for item in items:
            v = item.get_vector()
            logit = v[:-1] @ mle_theta + v[-1]
            denom = 2 + np.exp(logit) + np.exp(-logit)
            fisher_matrix += np.outer(v[:-1], v[:-1]) / denom

        self.F = fisher_matrix
        _, self.logdet = np.linalg.slogdet(self.F)

    def update_fisher_information_matrix(self, item_id):
        """Update the Fisher information matrix with a new item."""
        item = self._get_item(item_id)
        v = item.get_vector()
        mle_theta=self.mle_theta.copy() 
        logit = v[:-1] @ mle_theta + v[-1]
        denom = 2 + np.exp(logit) + np.exp(-logit)
        fisher_item = np.outer(v[:-1], v[:-1]) / denom

        self.F += fisher_item
        _, self.logdet = np.linalg.slog

    ### Fisher Information ###
    def fisher_info_item(self, item_id):
        """Compute the Fisher information for an item."""
        item = self._get_item(item_id)
        v = item.get_vector()
        mle_theta=self.mle_theta.copy()
        logit = v[:-1] @ mle_theta + v[-1]
        denom = 2 + np.exp(logit) + np.exp(-logit)

        return np.outer(v[:-1], v[:-1]) / denom

    def update_hidden_state_exploration_matrix(self, item_id):
        """Update the exploration matrix with a new item."""
        item = self._get_item(item_id)
        v = item.kcs
        self.exploration_hidden_state_matrix[item_id] = (
            self.exploration_hidden_state_matrix[item_id] + np.outer(v, v)
        )
        self.inv_exploration_hidden_state_matrix[
            item_id
        ] = self.inv_exploration_hidden_state_matrix[item_id] - np.outer(
            self.inv_exploration_hidden_state_matrix[item_id] @ v,
            v @ self.inv_exploration_hidden_state_matrix[item_id],
        ) / (
            1 + v @ self.inv_exploration_hidden_state_matrix[item_id] @ v
        )

    ### Reward Computation ###
    def expected_reward(self, item_id, exploration_parameter):
        """Generalized expected reward computation."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response(item)
        return reward * response

    def expected_reward_mle(self, item_id, exploration_parameter):
        """Expected reward with MLE exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_mle(item)
        return reward * response

    def expected_reward_ucb(self, item_id, exploration_parameter):
        """Expected reward with UCB exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_mle(item)
        exploration = exploration_parameter * self.item_exploration_value(item_id)
        return reward * response + exploration

    def expected_reward_fisher(self, item_id, exploration_parameter):
        """Expected reward with Fisher exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_mle(item)
        new_fisher = self.F + self.fisher_info_item(item_id)
        new_logdet = np.linalg.slogdet(new_fisher)[1]
        exploration = exploration_parameter * (new_logdet - self.logdet.copy())
        return reward * response + exploration

    def expected_reward_ts(self, item_id, exploration_parameter):
        """Expected reward with Thompson Sampling exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_bayesian(item)
        return reward * response

    def expected_reward_proxi(self, item_id, exploration_parameter):
        """Expected reward with approximation exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_proxi(item)
        return reward * response
    
    def expected_reward_irt(self,item_id, exploration_parameter):
        """Expected reward with IRT exploration."""
       
        new_fisher = self.F + self.fisher_info_item(item_id)
        new_logdet = np.linalg.slogdet(new_fisher)[1]
        return  new_logdet

    def expected_reward_from_last_hidden_state(self, item_id, exploration_parameter):
        """Expected reward with approximation exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response, last_hidden_state = self.expected_response_hidden_state(item)
        explo_matrix = self.inv_exploration_hidden_state_matrix[item_id].copy()
        return (
            reward * response
            + exploration_parameter
            * np.sqrt(last_hidden_state @ (explo_matrix @ last_hidden_state.T))[0, 0]
        )

    def expected_reward_ellipsoid(self, item_id, exploration_parameter):
        """Expected reward with ellipsoid exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_ellipsoid(item)
        exploration = exploration_parameter * self.item_exploration_value(item_id)
        return reward * response + exploration

    ### Item Selection ###
    def get_best_item(self, reward_method, exploration_parameter, epsilon, greedy):
        """Select the best item based on a reward method."""

        rewards = {
            a: reward_method(a, exploration_parameter=exploration_parameter)
            for a in self.usable_items_index
        }
        
        if greedy:
            return max(rewards, key=rewards.get)
        else:
            if np.random.uniform() < epsilon:
                return np.random.choice(list(rewards.keys()))
            else:
                return max(rewards, key=rewards.get)

    ### Optimization ###
    def optimize_prior(self):
        """Optimize the prior using MAP."""

        def objective(x):
            

            return -log_posterior_map(x, self.corpus, self.learning_trace, self.prior)

        init_param = self.mu.copy()

        res = minimize(objective, init_param,bounds=[(-5,5)]*self.theta_dim ,method="L-BFGS-B")

        a = res.x
        b = res.hess_inv.todense()
        self.mu = a.copy()
        self.sigma = np.array(b).copy()
        self.prior = multivariate_normal(mean=a, cov=b)
        self.prior_history.append(multivariate_normal(mean=a, cov=b))
        self.mu_history.append(a)
        self.sigma_history.append(b)
    def classical_optimization(self):
        """Optimize parameters using MLE."""

        def objective(x):
            x= np.clip(x, -10, 10)
            return -log_posterior(x, self.corpus, self.learning_trace)

        res = minimize(objective, np.zeros_like(self.theta), method="BFGS")
        self.mle_theta = res.x
        self.mle_theta_history.append(self.mle_theta.copy())

    def regularized_optimization(self):
        """Optimized parameter using reg MLE

        Returns:
            : reg MLE theta
        """

        def objective(x):
            return -reg_log_likelihood(
                x, self.corpus, self.learning_trace, self.lambda_
            )

        res = minimize(objective, np.zeros_like(self.theta), method="BFGS")
        self.res = res
        self.reg_mle_theta = res.x
        self.reg_mle_theta_history.append(self.reg_mle_theta.copy())
        self.H_t_inv = res.hess_inv

    def optmization_from_ellipsoid(self):
        """Optimize parameters using ellipsoid method."""

        def objective(x):          
            items_id, outcomes = zip(*self.learning_trace)
            return ellipsoid_over_items(
                x, self.reg_mle_theta, self.H_t_inv, self.corpus, items_id
            )

        res = minimize(objective, np.zeros_like(self.theta),bounds=[(-5,5)]*self.theta_dim, method="L-BFGS-B")
        self.ellipsoid_theta = res.x
        self.ellipsoid_theta_history.append(self.ellipsoid_theta.copy())

    ### Accuracy Metrics ###
    def accuracy(self, items, response_method):
        """Generalized accuracy computation."""
        true_responses = self.simulate_pluriel(items)
        predictions = np.array([response_method(item) for item in items])
        predicted_classes = (predictions > 0.5).astype(int)

        roc_auc = roc_auc_score(true_responses, predictions)
        mean_acc = accuracy_score(true_responses, predicted_classes)
        mse = np.mean((true_responses - predictions) ** 2)
        return roc_auc, mean_acc, mse

    def bandit_simulation(
        self,
        n_rounds,
        exploration_parameter,
        reward_method_name,
        epsilon,
        item_removal,
        start_round=0,
        dynamic_exploration=False,
    ):
        """Simulate the bandit problem."""
        reward_methods = {
            "expected_reward": self.expected_reward,
            "expected_reward_mle": self.expected_reward_mle,
            "expected_reward_ucb": self.expected_reward_ucb,
            "expected_reward_fisher": self.expected_reward_fisher,
            "expected_reward_ts": self.expected_reward_ts,
            "expected_reward_proxi": self.expected_reward_proxi,
            "expected_reward_from_last_hidden_state": self.expected_reward_from_last_hidden_state,
            "expected_reward_ellipsoid": self.expected_reward_ellipsoid,
            "epsilon_greedy": self.expected_reward_mle,
            "random_baseline": 0,
            "expected_reward_irt": self.expected_reward_irt,
        }
        reward_method = reward_methods[reward_method_name]

        # instantiate the exploration matrix or fisher_information matrix
        self.exploration_matrix()

        for t in range(start_round + 1, start_round + n_rounds + 1):

            if dynamic_exploration:
                dynamic_exploration_parameter = exploration_parameter * np.sqrt(np.log(t))
            elif not dynamic_exploration:

                dynamic_exploration_parameter = exploration_parameter

            if reward_method_name == "random_baseline":
                item_id = np.random.choice(self.usable_items_index)
                response = self.response(self._get_item(item_id))

            # do the optimization from the learning trace
            if (
                reward_method_name == "expected_reward_ts"
                or reward_method_name == "expected_reward_proxi"
            ):
                self.optimize_prior()
                self.sample_distribution()
                item_id = self.get_best_item(
                    reward_method,
                    exploration_parameter=dynamic_exploration_parameter,
                    epsilon=epsilon,
                    greedy=1,
                )
                response = self.response(self._get_item(item_id))

            if reward_method_name == "expected_reward_fisher":

                self.classical_optimization()
                self.fisher_information_matrix()
                item_id = self.get_best_item(
                    reward_method,
                    exploration_parameter=dynamic_exploration_parameter,
                    epsilon=epsilon,
                    greedy=1,
                )
                response = self.response(self._get_item(item_id))

            if reward_method_name == "expected_reward_irt":
                self.classical_optimization()
                self.fisher_information_matrix()
                item_id = self.get_best_item(
                    reward_method,
                    exploration_parameter=dynamic_exploration_parameter,
                    epsilon=epsilon,
                    greedy=1,
                )
                response = self.response(self._get_item(item_id))    

            if reward_method_name == "expected_reward_ellipsoid":
                self.regularized_optimization()
                self.optmization_from_ellipsoid()
                item_id = self.get_best_item(
                    reward_method,
                    exploration_parameter=dynamic_exploration_parameter,
                    epsilon=epsilon,
                    greedy=1,
                )
                response = self.response(self._get_item(item_id))
                self.update_exploration_matrix(item_id)
            if (
                reward_method_name == "expected_reward_mle"
                or reward_method_name == "expected_reward_ucb"
                or reward_method_name == "epsilon_greedy"
            ):

                self.classical_optimization()
                if reward_method_name == "epsilon_greedy":

                    item_id = self.get_best_item(
                        reward_method,
                        exploration_parameter=dynamic_exploration_parameter,
                        epsilon=epsilon,
                        greedy=0,
                    )
                else:

                    item_id = self.get_best_item(
                        reward_method,
                        exploration_parameter=dynamic_exploration_parameter,
                        epsilon=epsilon,
                        greedy=1,
                    )
                response = self.response(self._get_item(item_id))
                self.update_exploration_matrix(item_id)

            if reward_method_name == "expected_reward_from_last_hidden_state":
                item_id = self.get_best_item(
                    reward_method,
                    exploration_parameter=dynamic_exploration_parameter,
                    epsilon=epsilon,
                    greedy=1,
                )
                response = self.response(self._get_item(item_id))
                self.update_hidden_state_exploration_matrix(item_id)
                self.get_last_hidden_state()

            elif reward_method_name == "expected_reward":
                item_id = self.get_best_item(
                    reward_method,
                    exploration_parameter=dynamic_exploration_parameter,
                    epsilon=epsilon,
                    greedy=1,
                )
                response = self.response(self._get_item(item_id))

            best_item_id = self.get_best_item(
                self.expected_reward, exploration_parameter=0, epsilon=0, greedy=True
            )

            reward_item = self._get_reward(item_id)
            self.rewards_list.append(response * reward_item)
            a = self.expected_reward(item_id, dynamic_exploration_parameter).copy()
            b = self.expected_reward(best_item_id, dynamic_exploration_parameter).copy()
            self.expected_reward_list.append(a)
            self.regrets_list.append(b - a)
            if item_removal:

                self.usable_items_index.remove(item_id)

                