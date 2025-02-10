import os
import sys
import numpy as np
import argparse
import pandas as pd

import matplotlib.pyplot as plt

from student import Student
from dkt import *
from student import *
from bayes_classe import BayesianStudent
from optimizations import (
    log_posterior_map,
    log_posterior,
    reg_log_likelihood,
    ellipsoid_over_items,
)

base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)

def run_experiment(method_name, dim_theta, n_students, n_rounds, exploration_parameter,
                   lambda_, corpus, cold_start_len, item_removal,
                   knowledge_evolving_time_step, epsilon):
    """
    Runs one experiment for a single `method_name` and a single `exploration_parameter`.
    Returns a DataFrame with columns: [Round, Expected Reward, Regret, Real Reward].
    """


    # Initialize arrays to accumulate metrics
    avg_expected_rewards = np.zeros(n_rounds)
    avg_regrets = np.zeros(n_rounds)
    avg_real_rewards = np.zeros(n_rounds)

    # We can create some students for the cold start model or any needed pre-training
    # (If you need it for your approach).
    # But typically, we only need a new student for the actual simulation part:
    cold_start_students = {}
    for i in range(200):
        s = generate_student(dim_theta)
        s.simulate_pluriel(
            items=corpus.list_items(np.random.randint(0, corpus.nb_items, size=cold_start_len))
        )
        cold_start_students[i] = s

    # For each student in our simulation
    for _ in range(n_students):
        
        gen_student = generate_student(dim_theta)
        theta = gen_student.theta
        # BayesianStudent or whichever Student object you're using
        student = BayesianStudent(
            theta, 
            corpus=corpus, 
            mu=np.zeros(gen_student.theta_dim), 
            sigma=np.eye(gen_student.theta_dim), 
            lambda_=lambda_
        )

        # cold start for this single student
        student.simulate_pluriel(
            corpus.list_items(np.random.randint(0, corpus.nb_items, size=cold_start_len))
        )

        # If knowledge evolves every X timesteps:
        if knowledge_evolving_time_step > 0:
            # We break n_rounds into chunks of size knowledge_evolving_time_step
            chunks = n_rounds // knowledge_evolving_time_step
            for _ in range(chunks):
                student.bandit_simulation(
                    knowledge_evolving_time_step,
                    exploration_parameter=exploration_parameter,
                    reward_method_name=method_name,
                    lambda_=lambda_,
                    epsilon=epsilon,
                    item_removal=item_removal
                )
                # Then artificially "improve" the student's knowledge
                student.improvement(generate_learning_gains(gen_student.theta_dim))
        else:
            # No knowledge chunks, just do a single bandit simulation
            student.bandit_simulation(
                n_rounds,
                exploration_parameter=exploration_parameter,
                reward_method_name=method_name,
                epsilon=epsilon,
                item_removal=item_removal
            )

        # Accumulate the results
        avg_expected_rewards += np.array(student.expected_reward_list)
        avg_regrets += np.array(student.regrets_list)
        avg_real_rewards += np.array(student.rewards_list)

    # Average over all students
    avg_expected_rewards /= n_students
    avg_regrets /= n_students
    avg_real_rewards /= n_students

    # Build DataFrame
    df_results = pd.DataFrame({
        "Round": range(n_rounds),
        "Expected Reward": avg_expected_rewards,
        "Regret": avg_regrets,
        "Real Reward": avg_real_rewards
    })
    return df_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Plot the average metrics for a single reward method over n_rounds, \
                     across multiple exploration_parameter values (grid search).'
    )
    parser.add_argument('--dim_theta', type=int, default=10, help='Dimension of theta')
    parser.add_argument('--n_students', type=int, default=100, help='Number of students')
    parser.add_argument('--n_rounds', type=int, default=50, help='Number of rounds')
    parser.add_argument('--exploration_parameter', type=float, default=0.1,
                        help='(Deprecated) Single exploration parameter - not used directly, we do grid search now.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon parameter')
    parser.add_argument('--lambda_', type=float, default=0.01, help='Lambda parameter')
    parser.add_argument('--corpus_nb_items', type=int, default=50, help='Corpus object')
    parser.add_argument('--cold_start_len', type=int, default=5, help='Cold start length')
    parser.add_argument('--item_removal', type=bool, default=False, help='Item removal flag')
    parser.add_argument('--knowledge_evolving_time_step', type=int, default=0,
                        help='Knowledge evolving time step')
    parser.add_argument('--method', type=str, default='expected_reward',
                        help='Which single reward method to run. E.g. "expected_reward", "epsilon_greedy", etc.')

    args = parser.parse_args()
    # Construct your corpus
    corpus = Corpus(args.corpus_nb_items, args.dim_theta)

    # Define a grid of exploration parameters you want to try.
    # You can adjust this range or specific values as needed.
    exploration_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    # We'll store all results in a single DataFrame
    all_results = []

    for epar in exploration_grid:
        print(f"\nRunning method='{args.method}' with exploration_parameter={epar} ...")
        df_metrics = run_experiment(
            method_name=args.method,
            dim_theta=args.dim_theta,
            n_students=args.n_students,
            n_rounds=args.n_rounds,
            exploration_parameter=epar,
            lambda_=args.lambda_,
            corpus=corpus,
            cold_start_len=args.cold_start_len,
            item_removal=args.item_removal,
            knowledge_evolving_time_step=args.knowledge_evolving_time_step,
            epsilon=args.epsilon
        )
        # Tag each row with the exploration_parameter used
        df_metrics["exploration_parameter"] = epar
        all_results.append(df_metrics)

    # Concatenate all results
    df_all = pd.concat(all_results, ignore_index=True)
    # Save for future reference
    df_all.to_csv("grid_search_results.csv", index=False)

    # Now let's do the plotting:
    # We'll group by exploration_parameter and plot the average curves of:
    # 'Expected Reward', 'Regret', 'Real Reward' vs. Round
    plt.figure(figsize=(12, 8))

    # We will have 3 subplots: 1) Expected Reward, 2) Regret, 3) Real Reward
    # Each subplot will have multiple lines (one for each exploration param).
    metrics = ["Expected Reward", "Regret", "Real Reward"]
    for idx, metric in enumerate(metrics, start=1):
        plt.subplot(3, 1, idx)
        for epar in exploration_grid:
            subset = df_all[df_all["exploration_parameter"] == epar]
            plt.plot(subset["Round"], subset[metric], label=f"{metric} (exploration={epar})")
        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.legend()

    plt.suptitle(f"Method: {args.method} | Grid Search over exploration_parameter")
    plt.tight_layout()
    plt.savefig("grid_search_plot.pdf")
    

    print("\nDone! The results are in 'grid_search_results.csv' and the plot is 'grid_search_plot.pdf'.")
