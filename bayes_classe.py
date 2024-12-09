import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np


import matplotlib.pyplot as plt
from student import Student

from dkt import *

from optimizations import log_posterior_map, log_posterior


class BayesianStudent(Student):
    def __init__(self, theta, mu, sigma, corpus):
        super().__init__(theta)
        self.mu = mu
        self.sigma = sigma
        self.prior = multivariate_normal(mean=mu, cov=sigma)
        self.corpus = corpus
        
        # Histories
        self.mu_history = [mu]
        self.sigma_history = [sigma]
        self.prior_history = [self.prior]
        self.sampled_theta_history = [mu]
        self.mle_theta_history = []
        
        # Logs
        self.rewards_list = []
        self.regrets_list = []
        self.expected_reward_list = []

    ### Utility Methods ###
    def _get_item_vector(self, item):
        """Create the vector representation of an item."""
        return np.r_[item.kcs, -item.difficulty]

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

    def get_trained_model(self,trained_model,item_hidden_states, item_outcomes):
        self.trained_model=trained_model
        self.item_hidden_states=item_hidden_states
        self.item_outcomes=item_outcomes


    def get_fitted_models(self,classifier):
        self.classifier=classifier
        self.fitted_models=get_fitted_models(self.classifier,self.item_hidden_states, self.item_outcomes)


    def get_last_hidden_state(self):
        self.encoded_learning_trace,_ = preprocess_learning_trace(self.learning_trace, self.corpus.nb_items)
        _,(a,_)=self.trained_model.rnn(self.encoded_learning_trace)
        self.last_hidden_state=a.detach().numpy()
    ### Expected Responses ###


    def expected_response(self, item):
        """Expected correctness probability using real theta."""
        theta=self.theta.copy()
        return sigmoid(item.kcs @ theta-item.difficulty)
    def expected_response_bayesian(self, item):
        """Expected correctness probability using sampled theta."""
        sampled_theta = self.sample_distribution()
        return sigmoid(item.kcs @ sampled_theta-item.difficulty)

    def expected_response_mle(self, item):
        """Expected correctness probability using MLE theta."""
        mle_theta=self.mle_theta.copy()
        return sigmoid(item.kcs @ mle_theta-item.difficulty)

    def expected_response_proxi(self, item):
        """Expected correctness probability using approximation."""
        item_vector = self._get_item_vector(item)
        mu_a = self.mu @ item_vector[:-1] - item_vector[-1]
        sigma_a = item_vector[:-1] @ (self.sigma @ item_vector[:-1])
        return sigmoid(mu_a / np.sqrt(1 + np.pi * sigma_a / 8))
    
    

    ### Exploration Matrix ###
    def exploration_matrix(self, lambda_):
        """Compute the exploration matrix."""
        items_id, _ = zip(*self.learning_trace)
        items = [self._get_item(item_id) for item_id in items_id]
        v_matrix = np.array([self._get_item_vector(item) for item in items])
        
        H = lambda_ * np.eye(len(self.theta) + 1) + v_matrix.T @ v_matrix
        self.V = H
        self.inv_V = np.linalg.inv(H)  # Cached for reuse

    def item_exploration_value(self, item_id):
        """Compute the exploration value of an item."""
        item = self._get_item(item_id)
        v = self._get_item_vector(item)
        return np.sqrt(v @ self.inv_V @ v)


    def fisher_information_matrix(self):
        """Compute the Fisher information matrix."""
        items_id, _ = zip(*self.learning_trace)
        items = [self._get_item(item_id) for item_id in items_id]
        fisher_matrix = np.zeros((len(self.theta) + 1, len(self.theta) + 1))

        for item in items:
            v = self._get_item_vector(item)
            logit = v[:-1] @ self.mle_theta - v[-1]
            denom = 2 + np.exp(logit) + np.exp(-logit)
            fisher_matrix += np.outer(v, v) / denom

        self.F= fisher_matrix
        _, self.logdet = np.linalg.slogdet(self.F)



    ### Fisher Information ###
    def fisher_info_item(self, item_id):
        """Compute the Fisher information for an item."""
        item = self._get_item(item_id)
        v = self._get_item_vector(item)
        logit = v[:-1] @ self.mle_theta - v[-1]
        denom = 2 + np.exp(logit) + np.exp(-logit)

        return np.outer(v, v) / denom

    ### Reward Computation ###
    def expected_reward(self, item_id,exploration_parameter):
        """Generalized expected reward computation."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response=self.expected_response(item)
        return reward * response 
    
    def expected_reward_mle(self, item_id,exploration_parameter):
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

    def expected_reward_fisher(self, item_id,exploration_parameter):
        """Expected reward with Fisher exploration."""
        item = self._get_item(item_id)
        reward = self._get_reward(item_id)
        response = self.expected_response_mle(item)
        new_fisher=self.F + self.fisher_info_item(item_id)
        new_logdet = np.linalg.slogdet(new_fisher)[1]
        exploration = exploration_parameter * (new_logdet-self.logdet.copy())
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
    
    def expected_reward_from_last_hidden_state(self, item_id, exploration_parameter):
        """Expected reward with approximation exploration."""
        
        reward = self._get_reward(item_id)
        last_hidden_state=self.last_hidden_state.copy()
        response = self.fitted_models[item_id].predict_proba(last_hidden_state.reshape(1,-1))[0][1]
        return reward * response


    ### Item Selection ###
    def get_best_item(self, reward_method,exploration_parameter,epsilon,greedy):
        """Select the best item based on a reward method."""
        items = self.corpus.list_items(range(self.corpus.nb_items))
        rewards = {item.id: reward_method(item.id, exploration_parameter=exploration_parameter) for item in items}
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
            x = np.clip(x, -10, 10)

            return -log_posterior_map(x, self.corpus, self.learning_trace, self.prior)
        
        init_param=self.mu.copy()

        res = minimize(objective, init_param, method="BFGS")

        a = res.x
        b=res.hess_inv
        self.mu=a.copy()
        self.sigma = b.copy()
        self.prior = multivariate_normal(mean=a, cov=b)
        self.prior_history.append(multivariate_normal(mean=a, cov=b))
        self.mu_history.append(a)
        self.sigma_history.append(b)

    def classical_optimization(self):
        """Optimize parameters using MLE."""
        def objective(x):
            x = np.clip(x, -10, 10)
            return -log_posterior(x, self.corpus, self.learning_trace)

        res = minimize(objective, np.zeros_like(self.theta), method="BFGS")
        self.mle_theta = res.x
        self.mle_theta_history.append(self.mle_theta.copy())

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



    def bandit_simulation(self, n_rounds, exploration_parameter, reward_method_name, lambda_,epsilon,greedy):
        """Simulate the bandit problem."""
        reward_methods = {
            'expected_reward': self.expected_reward,
            'expected_reward_mle': self.expected_reward_mle,
            'expected_reward_ucb': self.expected_reward_ucb,
            'expected_reward_fisher': self.expected_reward_fisher,
            'expected_reward_ts': self.expected_reward_ts,
            'expected_reward_proxi': self.expected_reward_proxi,
            'expected_reward_from_last_hidden_state': self.expected_reward_from_last_hidden_state
        }
        reward_method = reward_methods[reward_method_name]
        for _ in range(n_rounds):

            # do the optimization from the learning trace
            self.optimize_prior()
            self.classical_optimization()

            # compute the fisher information matrix or the exploration matrix
            self.fisher_information_matrix()
            self.exploration_matrix(lambda_=lambda_)
            
            self.get_last_hidden_state()
            
            print([self.expected_reward_proxi(item_id=j,exploration_parameter=0.5) for j in range(100)])
            item_id = self.get_best_item(reward_method, exploration_parameter=exploration_parameter,epsilon=epsilon,greedy=greedy)
            print(item_id)
            best_item_id= self.get_best_item(self.expected_reward, exploration_parameter=0,epsilon=0,greedy=True)

            response=self.response(self._get_item(item_id))
            reward_item= self._get_reward(item_id)
            self.rewards_list.append( response * reward_item)
            a=self.expected_reward(item_id,exploration_parameter).copy()
            b=self.expected_reward(best_item_id,exploration_parameter).copy()
            self.expected_reward_list.append(a)
            self.regrets_list.append(b-a)
            




    def plot_metrics(self):
        """Plot the expected reward list, regret list, and real reward list."""
        rounds = range(len(self.rewards_list))

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(rounds, self.expected_reward_list, label='Expected Reward')
        plt.xlabel('Rounds')
        plt.ylabel('Expected Reward')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(rounds, self.regrets_list, label='Regret', color='orange')
        plt.xlabel('Rounds')
        plt.ylabel('Regret')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(rounds, self.rewards_list, label='Real Reward', color='green')
        plt.xlabel('Rounds')
        plt.ylabel('Real Reward')
        plt.legend()

        plt.tight_layout()
        plt.show()




        
def plot_average_metrics(students, n_rounds):
    
    """Plot the average metrics of n students over n_rounds for different reward methods."""
    reward_methods = {
        'Expected Reward': 'expected_reward',
        'Expected Reward MLE': 'expected_reward_mle',
        'Expected Reward UCB': 'expected_reward_ucb',
        'Expected Reward Fisher': 'expected_reward_fisher'
    }

    for method_name, method in reward_methods.items():
        avg_expected_rewards = np.zeros(n_rounds)
        avg_regrets = np.zeros(n_rounds)
        avg_real_rewards = np.zeros(n_rounds)

        for student in students:
            student.bandit_simulation(n_rounds, exploration_parameter=1, reward_method=getattr(student, method), lambda_=1, epsilon=0.1, greedy=False)
            avg_expected_rewards += np.array(student.expected_reward_list)
            avg_regrets += np.array(student.regrets_list)
            avg_real_rewards += np.array(student.rewards_list)

        avg_expected_rewards /= len(students)
        avg_regrets /= len(students)
        avg_real_rewards /= len(students)

        rounds = range(n_rounds)

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(rounds, avg_expected_rewards, label=f'Average Expected Reward ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Expected Reward')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(rounds, avg_regrets, label=f'Average Regret ({method_name})', color='orange')
        plt.xlabel('Rounds')
        plt.ylabel('Average Regret')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(rounds, avg_real_rewards, label=f'Average Real Reward ({method_name})', color='green')
        plt.xlabel('Rounds')
        plt.ylabel('Average Real Reward')
        plt.legend()

        plt.tight_layout()
        plt.show()