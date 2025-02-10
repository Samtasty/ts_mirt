import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse
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
from bayes_classe import BayesianStudent
import pandas as pd

from optimizations import (
    log_posterior_map,
    log_posterior,
    reg_log_likelihood,
    ellipsoid_over_items,
)
import matplotlib.pyplot as plt

base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)

def plot_average_metrics(dim_theta, n_students, n_rounds, exploration_parameter, lambda_, corpus, cold_start_len, item_removal, knowledge_evolving_time_step,epsilon):
    """Plot the average metrics of n students over n_rounds for different reward methods."""
    reward_methods = {
        'Expected Reward': 'expected_reward',
        'Expected Reward MLE': 'expected_reward_mle',
        "Epsilon Greedy": 'epsilon_greedy',
        'Expected Reward UCB': 'expected_reward_ucb',
        'Expected Reward Fisher': 'expected_reward_fisher',
        'Expected Reward Thompson': 'expected_reward_ts',
        'expected Reward Proxi': 'expected_reward_proxi',
        "expected Reward Ellipsoid": 'expected_reward_ellipsoid',
        
    }

    avg_metrics = {method_name: {'expected_rewards': np.zeros(n_rounds),
                                 'regrets': np.zeros(n_rounds),
                                 'real_rewards': np.zeros(n_rounds)}
                   for method_name in reward_methods.keys()}
    
    dic_students = {i: generate_student(dim_theta) for i in range(200)}

    for i in range(200):
        dic_students[i].simulate_pluriel(items=corpus.list_items(np.random.randint(0, corpus.nb_items, size=cold_start_len)))

    #model = DKT(input_dim=corpus.nb_items*2, hidden_dim=5, output_dim=corpus.nb_items)
    #new_trained_model, item_hidden_states, item_outcomes = training_model(model, dic_students, corpus=corpus, epochs=20, batch_size=16, learning_rate=0.01)

    for method_name, method in reward_methods.items():
        print('method_name', method_name)
        for _ in range(n_students):
            gen_student = generate_student(dim_theta)
            theta = gen_student.theta
                    
            student = BayesianStudent(theta, corpus=corpus, mu=np.zeros(gen_student.theta_dim), sigma=np.eye(gen_student.theta_dim),lambda_=lambda_)
            student.simulate_pluriel(corpus.list_items(np.random.randint(0, corpus.nb_items, size=cold_start_len)))
            # student.get_trained_model(new_trained_model, item_hidden_states, item_outcomes)
            # student.get_fitted_models(LogisticRegression())
            if knowledge_evolving_time_step > 0:
                for _ in range(n_rounds // knowledge_evolving_time_step):
                    student.bandit_simulation(knowledge_evolving_time_step, exploration_parameter=exploration_parameter, reward_method_name=method, epsilon=epsilon, item_removal=item_removal)
                    student.improvement(generate_learning_gains(gen_student.theta_dim))
            else:
                student.bandit_simulation(n_rounds, exploration_parameter=exploration_parameter, reward_method_name=method, epsilon=epsilon, item_removal=item_removal)
            
            avg_metrics[method_name]['expected_rewards'] += np.array(student.expected_reward_list)
            avg_metrics[method_name]['regrets'] += np.array(student.regrets_list)
            avg_metrics[method_name]['real_rewards'] += np.array(student.rewards_list)

        avg_metrics[method_name]['expected_rewards'] /= n_students
        avg_metrics[method_name]['regrets'] /= n_students
        avg_metrics[method_name]['real_rewards'] /= n_students

    rounds = range(n_rounds)

    plt.figure(figsize=(12, 8))

    for method_name in reward_methods.keys():
        plt.subplot(3, 1, 1)
        plt.plot(rounds, avg_metrics[method_name]['expected_rewards'], label=f'Average Expected Reward ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Expected Reward')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(rounds, avg_metrics[method_name]['regrets'], label=f'Average Regret ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Regret')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(rounds, avg_metrics[method_name]['real_rewards'], label=f'Average Real Reward ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Real Reward')
        plt.legend()

    plt.tight_layout()
    plt.savefig('regret_plot.pdf')
    #plt.show()
        # Convert avg_metrics to DataFrame
    rows = []

    for method_name in reward_methods.keys():
        for round_num in range(n_rounds):
            rows.append({
                'Method': method_name,
                'Round': round_num,
                'Expected Reward': avg_metrics[method_name]['expected_rewards'][round_num],
                'Regret': avg_metrics[method_name]['regrets'][round_num],
                'Real Reward': avg_metrics[method_name]['real_rewards'][round_num]
            })

    df_metrics = pd.DataFrame(rows)

    df_metrics.to_csv('avg_metrics.csv', index=False)
    return df_metrics
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot the average metrics of n students over n_rounds for different reward methods.')
    parser.add_argument('--dim_theta', type=int, default=10, help='Dimension of theta')
    parser.add_argument('--n_students', type=int, default=100, help='Number of students')
    parser.add_argument('--n_rounds', type=int, default=50, help='Number of rounds')
    parser.add_argument('--exploration_parameter', type=float, default=0.1, help='Exploration parameter')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon parameter')
    parser.add_argument('--lambda_', type=float, default=0.01, help='Lambda parameter')
    parser.add_argument('--corpus_nb_items', type=int, default=50, help='Corpus object')
    #parser.add_argument('--corpus_kcs_dim', type=int, default=10, help='Corpus object')
    parser.add_argument('--cold_start_len', type=int, default=5, help='Cold start length')
    parser.add_argument('--item_removal', type=bool, default=False, help='Item removal flag')
    parser.add_argument('--knowledge_evolving_time_step', type=int, default=0, help='Knowledge evolving time step')


    args = parser.parse_args()
    corpus=Corpus(args.corpus_nb_items, args.dim_theta)

    plot_average_metrics(
        dim_theta=args.dim_theta,
        n_students=args.n_students,
        n_rounds=args.n_rounds,
        exploration_parameter=args.exploration_parameter,
        lambda_=args.lambda_,
        corpus=corpus,
        cold_start_len=args.cold_start_len,
        item_removal=args.item_removal,
        knowledge_evolving_time_step=args.knowledge_evolving_time_step,
        epsilon=args.epsilon
    )

