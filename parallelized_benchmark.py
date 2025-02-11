from joblib import Parallel, delayed
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




def simulate_one_student(
    dim_theta,
    n_rounds, 
    method_name,  
    exploration_parameter,
    corpus,
    cold_start_len,
    knowledge_evolving_time_step,
    epsilon,
    item_removal,
    lambda_
):
    """
    Simulate a single student for the given bandit method, returning
    lists (or arrays) of expected rewards, regrets, real rewards.
    """

    # 1) Generate random student
    gen_student = generate_student(dim_theta)
    theta = gen_student.theta
    
    # 2) Create your BayesianStudent (or whichever class)
    student = BayesianStudent(
        theta,
        corpus=corpus,
        mu=np.zeros(gen_student.theta_dim),
        sigma=np.eye(gen_student.theta_dim),
        lambda_=lambda_
    )



    
    # 3) Pre-simulate "cold start"
    student.simulate_pluriel(
        corpus.list_items(np.random.randint(0, corpus.nb_items, size=cold_start_len))
    )

    # 4) Actually run bandit simulation
    if knowledge_evolving_time_step > 0:
        # Repeated chunk: do part of simulation, then student "improvement"
        for _ in range(n_rounds // knowledge_evolving_time_step):
            student.bandit_simulation(
                knowledge_evolving_time_step, 
                exploration_parameter=exploration_parameter,
                reward_method_name=method_name, 
                epsilon=epsilon, 
                item_removal=item_removal
            )
            student.improvement(generate_learning_gains(gen_student.theta_dim))
    else:
        # single chunk
        student.bandit_simulation(
            n_rounds,
            exploration_parameter=exploration_parameter,
            reward_method_name=method_name,
            epsilon=epsilon,
            item_removal=item_removal
        )
    
    # 5) Return the arrays of interest
    return (
        np.array(student.expected_reward_list),
        np.array(student.regrets_list),
        np.array(student.rewards_list)
    )



def plot_average_metrics(dim_theta, n_students, n_rounds, exploration_parameter, lambda_, corpus, cold_start_len, item_removal, knowledge_evolving_time_step, epsilon):
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

    avg_metrics = {
        method_name: {
            'expected_rewards': np.zeros(n_rounds),
            'regrets': np.zeros(n_rounds),
            'real_rewards': np.zeros(n_rounds),
            'cumulated_expected_rewards': np.zeros(n_rounds),
            'cumulated_regrets': np.zeros(n_rounds),
            'cumulated_real_rewards': np.zeros(n_rounds)
        }
        for method_name in reward_methods.keys()
    }
    #%%
    # OPTIONAL: if you truly need these outside or for some reason
    # dic_students = {i: generate_student(dim_theta) for i in range(200)}
    # for i in range(200):
    #     dic_students[i].simulate_pluriel(...)
    #
    # do that if needed, or remove it if not used

    for method_name, method in reward_methods.items():
        print('method_name', method_name)

        # We parallelize the simulation over N students:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(simulate_one_student)(
                dim_theta=dim_theta,
                n_rounds=n_rounds,
                method_name=method,
                exploration_parameter=exploration_parameter,
                corpus=corpus,
                cold_start_len=cold_start_len,
                knowledge_evolving_time_step=knowledge_evolving_time_step,
                epsilon=epsilon,
                item_removal=item_removal,
                lambda_=lambda_
            )
            for _ in range(n_students)
        )

        # "results" is a list of tuples (exp_r, regrets, real_r), one per student
        for (exp_r, rg, real_r) in results:
            avg_metrics[method_name]['expected_rewards'] += exp_r
            avg_metrics[method_name]['cumulated_expected_rewards'] += np.cumsum(exp_r)
            avg_metrics[method_name]['regrets'] += rg
            avg_metrics[method_name]['cumulated_regrets'] += np.cumsum(rg)
            avg_metrics[method_name]['real_rewards'] += real_r
            avg_metrics[method_name]['cumulated_real_rewards'] += np.cumsum(real_r)

        # Divide by n_students to get the average
        avg_metrics[method_name]['expected_rewards'] /= n_students
        avg_metrics[method_name]['cumulated_expected_rewards'] /= n_students
        avg_metrics[method_name]['regrets'] /= n_students
        avg_metrics[method_name]['cumulated_regrets'] /= n_students
        avg_metrics[method_name]['real_rewards'] /= n_students
        avg_metrics[method_name]['cumulated_real_rewards'] /= n_students

    # PLOTTING (unchanged)
    rounds = range(n_rounds)
    plt.figure(figsize=(12, 8))

    for method_name in reward_methods.keys():
        plt.subplot(3, 1, 1)
        plt.plot(rounds, avg_metrics[method_name]['cumulated_expected_rewards'], label=f'Avg Expected Reward ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Expected Reward')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(rounds, avg_metrics[method_name]['cumulated_regrets'], label=f'Average Regret ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Regret')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(rounds, avg_metrics[method_name]['cumulated_real_rewards'], label=f'Average Real Reward ({method_name})')
        plt.xlabel('Rounds')
        plt.ylabel('Average Real Reward')
        plt.legend()

    plt.tight_layout()
    plt.savefig('regret_plot_paral.pdf')

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
    df_metrics.to_csv('avg_metrics_paral.csv', index=False)
    return df_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the average metrics of n students over n_rounds for different reward methods.')
    parser.add_argument('--dim_theta', type=int, default=10, help='Dimension of theta')
    parser.add_argument('--n_students', type=int, default=100, help='Number of students')
    parser.add_argument('--n_rounds', type=int, default=50, help='Number of rounds')
    parser.add_argument('--exploration_parameter', type=float, default=0.1, help='Exploration parameter')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon parameter')
    parser.add_argument('--lambda_', type=float, default=0.01, help='Lambda parameter')
    parser.add_argument('--corpus_nb_items', type=int, default=50, help='Number of items in the corpus')
    parser.add_argument('--cold_start_len', type=int, default=5, help='Cold start length')
    parser.add_argument('--item_removal', type=bool, default=False, help='Item removal flag')
    parser.add_argument('--knowledge_evolving_time_step', type=int, default=0, help='Knowledge evolving time step')

    args = parser.parse_args()

    # Construct our corpus
    corpus = Corpus(nb_items=args.corpus_nb_items, kcs_dim=args.dim_theta)

    # Run the parallel simulation + plotting
    df_metrics = plot_average_metrics(
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

    print("\nSimulation completed. Results saved to 'avg_metrics.csv' and 'regret_plot.pdf'.") 
# %%
