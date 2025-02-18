import os
import sys
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed  # Parallel computing

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


def simulate_one_student(
    l,
    dim_theta,
    n_rounds,
    exploration_parameter,
    lambda_,
    corpus,
    cold_start_len,
    item_removal,
    knowledge_evolving_time_step,
    epsilon,
    reward_methods,
    dynamic_exploration,
):
    """Simulate a single student and return results for each method."""
    
    gen_student = generate_student(dim_theta)
    initial_theta = gen_student.theta.copy()
    student = BayesianStudent(
        initial_theta,
        corpus=corpus,
        mu=np.zeros(gen_student.theta_dim),
        sigma=np.eye(gen_student.theta_dim),
        lambda_=lambda_,
    )
    student.simulate_pluriel(
        corpus.list_items(np.random.randint(0, corpus.nb_items, size=cold_start_len))
    )
    starting_learning_trace = student.learning_trace.copy()

    student_results = {
        method_name: {
            "expected_rewards": np.zeros(n_rounds),
            "regrets": np.zeros(n_rounds),
            "real_rewards": np.zeros(n_rounds),
            "cumulated_expected_rewards": np.zeros(n_rounds),
            "cumulated_regrets": np.zeros(n_rounds),
            "cumulated_real_rewards": np.zeros(n_rounds),
        }
        for method_name in reward_methods.keys()
    }

    for method_name, method in reward_methods.items():
        init_theta = initial_theta.copy()
        student = BayesianStudent(
            init_theta,
            corpus=corpus,
            mu=np.zeros(gen_student.theta_dim),
            sigma=np.eye(gen_student.theta_dim),
            lambda_=lambda_,
        )
        student.learning_trace = starting_learning_trace.copy()
        
        if knowledge_evolving_time_step > 0:
            # Simulate in chunks with dynamic exploration enabled
            for m in range(n_rounds // knowledge_evolving_time_step):
                student.bandit_simulation(
                    knowledge_evolving_time_step,
                    exploration_parameter=exploration_parameter,
                    reward_method_name=method,
                    epsilon=epsilon,
                    item_removal=item_removal,
                    dynamic_exploration=dynamic_exploration
                )
                learning_gains = generate_learning_gains(gen_student.theta_dim)
                student.improvement(learning_gains)
            # Leftover rounds if n_rounds is not divisible by knowledge_evolving_time_step
            leftover = n_rounds % knowledge_evolving_time_step
            if leftover > 0:
                student.bandit_simulation(
                    leftover,
                    exploration_parameter=exploration_parameter,
                    reward_method_name=method,
                    epsilon=epsilon,
                    item_removal=item_removal,
                    dynamic_exploration=dynamic_exploration
                )
        else:
            # Single chunk simulation with dynamic exploration
            student.bandit_simulation(
                n_rounds,
                exploration_parameter=exploration_parameter,
                reward_method_name=method,
                epsilon=epsilon,
                item_removal=item_removal,
                dynamic_exploration=dynamic_exploration
            )

        student_results[method_name]["expected_rewards"] = np.array(student.expected_reward_list)
        student_results[method_name]["regrets"] = np.array(student.regrets_list)
        student_results[method_name]["real_rewards"] = np.array(student.rewards_list)
        student_results[method_name]["cumulated_expected_rewards"] = np.cumsum(student.expected_reward_list)
        student_results[method_name]["cumulated_regrets"] = np.cumsum(student.regrets_list)
        student_results[method_name]["cumulated_real_rewards"] = np.cumsum(student.rewards_list)

    return student_results


def run_experiment_parallel(
    
    dim_theta,
    n_students,
    n_rounds,
    exploration_parameter,
    lambda_,
    corpus,
    cold_start_len,
    item_removal,
    knowledge_evolving_time_step,
    epsilon,
    reward_methods,
    dynamic_exploration
):
    """
    Run the experiment in parallel over n_students and return a DataFrame.
    The DataFrame contains columns: [Round, Method, Cumulative Regret].
    """
    reward_methods = {
        "Optimal Policy": "expected_reward",
        "Greedy Algorithm": "expected_reward_mle",
        "Epsilon-Greedy": "epsilon_greedy",
        "GLM-UCB": "expected_reward_ucb",
        "Fisher Information Exploration": "expected_reward_fisher",
        "Classical Thompson Sampling": "expected_reward_ts",
        "Direct Bayesian Inference": "expected_reward_proxi",
        "Improved UCB": "expected_reward_ellipsoid",
        "Random Baseline": "random_baseline",
        "IRT decision rule": "expected_reward_irt",
    }

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(simulate_one_student)(
            l,
            dim_theta,
            n_rounds,
            exploration_parameter,
            lambda_,
            corpus,
            cold_start_len,
            item_removal,
            knowledge_evolving_time_step,
            epsilon,
            reward_methods,
            dynamic_exploration
        )
        for l in range(n_students)
    )

    # Build a DataFrame from the cumulated regret values.
    # We'll assume rounds are 1-indexed for clarity.
    rows = []
    # Define the desired rounds at which to record the cumulative regret.
    desired_rounds = [5, 10, 20, 30]
    for student_result in results:
        for method in reward_methods.keys():
            cumulated_regrets = student_result[method]["cumulated_regrets"]
            # Note: our simulation produces an array of length n_rounds.
            # We assume that n_rounds >= max(desired_rounds)
            for r in desired_rounds:
                # Adjust for 0-indexing: cumulative regret at round r is at index r-1.
                rows.append({
                    "Method": method,
                    "Round": r,
                    "Cumulative Regret": cumulated_regrets[r - 1]
                })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments over various knowledge evolution and item removal settings, "
                    "and create a CSV file comparing the cumulative regret at rounds 5, 10, 20, and 30."
    )
    parser.add_argument("--dim_theta", type=int, default=10, help="Dimension of theta")
    parser.add_argument("--n_students", type=int, default=100, help="Number of students")
    parser.add_argument("--n_rounds", type=int, default=30, help="Number of rounds (must be >= 30)")
    parser.add_argument("--exploration_parameter", type=float, default=0.1, help="Exploration parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon parameter")
    parser.add_argument("--lambda_", type=float, default=0.01, help="Lambda parameter")
    parser.add_argument("--corpus_nb_items", type=int, default=50, help="Number of items in corpus")
    parser.add_argument("--dynamic_exploration", type=int, default=0, help="Dynamic exploration flag (0 or 1)")
    parser.add_argument("--cold_start_len", type=int, default=5, help="Cold start length")
    parser.add_argument("--item_removal", type=int, default=0, help="Item removal flag (0 or 1)")
    parser.add_argument("--knowledge_evolving_time_step", type=int, default=0, help="Knowledge evolving time step")

    args = parser.parse_args()
    corpus = Corpus(args.corpus_nb_items, args.dim_theta)

    # Run experiment for a given method over grid settings.
    # For demonstration, you can choose a single method (e.g., expected_reward).
    # Here, we loop over a grid for knowledge evolution and item removal.
    k_values = [0, 5, 2]
    removal_values = [0, 1]

    comparison_rows = []

    reward_methods = {
        "Optimal Policy": "expected_reward",
        "Greedy Algorithm": "expected_reward_mle",
        "Epsilon-Greedy": "epsilon_greedy",
        "GLM-UCB": "expected_reward_ucb",
        "Fisher Information Exploration": "expected_reward_fisher",
        "Classical Thompson Sampling": "expected_reward_ts",
        "Direct Bayesian Inference": "expected_reward_proxi",
        "Improved UCB": "expected_reward_ellipsoid",
        "Random Baseline": "random_baseline",
        "IRT decision rule": "expected_reward_irt",
    }


    for k in k_values:
        for removal in removal_values:
            print(f"Running experiment with knowledge_evolving_time_step={k} and item_removal={removal} ...")
            df_metrics = run_experiment_parallel(
                dim_theta=args.dim_theta,
                n_students=args.n_students,
                n_rounds=args.n_rounds,
                exploration_parameter=args.exploration_parameter,
                lambda_=args.lambda_,
                corpus=corpus,
                cold_start_len=args.cold_start_len,
                item_removal=removal,
                knowledge_evolving_time_step=k,
                epsilon=args.epsilon,
                reward_methods=reward_methods,
                dynamic_exploration=args.dynamic_exploration
            )
            # For each method, extract the cumulative regret at the desired rounds.
            desired_rounds = [5, 10, 20, 30]
            for method in df_metrics["Method"].unique():
                for r in desired_rounds:
                    value = df_metrics[(df_metrics["Method"] == method) & (df_metrics["Round"] == r)]["Cumulative Regret"].iloc[0]
                    comparison_rows.append({
                        "Method": method,
                        "knowledge_evolving_time_step": k,
                        "item_removal": removal,
                        "Round": r,
                        "Cumulative Regret": value
                    })

    # Create a comparison DataFrame and save to CSV
    df_comparison = pd.DataFrame(comparison_rows)
    csv_filename = f"table_cumulated_regret_dim_{args.dim_theta}_nc_{args.corpus_nb_items}.csv"
    df_comparison.to_csv(csv_filename, index=False)
    print(f"Comparison CSV saved as {csv_filename}")
