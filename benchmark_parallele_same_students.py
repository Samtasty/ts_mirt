import os
import sys
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
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
    intial_theta = gen_student.theta.copy()
    student = BayesianStudent(
        intial_theta,
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
        init_theta=intial_theta.copy()
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
                    dynamic_exploration=dynamic_exploration  # Added here
                )
                learning_gains=generate_learning_gains(gen_student.theta_dim)
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
                    dynamic_exploration=dynamic_exploration  # Added here
                )
        else:
            # Single chunk simulation with dynamic exploration
            student.bandit_simulation(
                n_rounds,
                exploration_parameter=exploration_parameter,
                reward_method_name=method,
                epsilon=epsilon,
                item_removal=item_removal,
                dynamic_exploration=dynamic_exploration  # Already present here
            )
        
        student_results[method_name]["expected_rewards"] = np.array(student.expected_reward_list)
        student_results[method_name]["regrets"] = np.array(student.regrets_list)
        student_results[method_name]["real_rewards"] = np.array(student.rewards_list)
        student_results[method_name]["cumulated_expected_rewards"] = np.cumsum(student.expected_reward_list)
        student_results[method_name]["cumulated_regrets"] = np.cumsum(student.regrets_list)
        student_results[method_name]["cumulated_real_rewards"] = np.cumsum(student.rewards_list)

    return student_results


def plot_average_metrics(
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
    dynamic_exploration
):
    """Parallelized computation of average metrics for different reward methods."""

    reward_methods = {
        "Optimal Policy": "expected_reward",
        "Greedy Algorithm":"expected_reward_mle",
        "Epsilon-Greedy": "epsilon_greedy",
        "GLM-UCB": "expected_reward_ucb",
        "Fisher Information Exploration": "expected_reward_fisher",
        "Classical Thompson Sampling": "expected_reward_ts",
        "Direct Bayesian Inference": "expected_reward_proxi",
        "Improved UCB": "expected_reward_ellipsoid",
        "Random Baseline": "random_baseline",
        "IRT decision rule": "expected_reward_irt"
    }

    # Parallel execution across students
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

    # Aggregate results across students
    avg_metrics = {
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

    for student_result in results:
        for method_name in reward_methods.keys():
            avg_metrics[method_name]["expected_rewards"] += student_result[method_name]["expected_rewards"]
            avg_metrics[method_name]["regrets"] += student_result[method_name]["regrets"]
            avg_metrics[method_name]["real_rewards"] += student_result[method_name]["real_rewards"]
            avg_metrics[method_name]["cumulated_expected_rewards"] += student_result[method_name]["cumulated_expected_rewards"]
            avg_metrics[method_name]["cumulated_regrets"] += student_result[method_name]["cumulated_regrets"]
            avg_metrics[method_name]["cumulated_real_rewards"] += student_result[method_name]["cumulated_real_rewards"]

    # Compute average
    for method_name in reward_methods.keys():
        avg_metrics[method_name]["expected_rewards"] /= n_students
        avg_metrics[method_name]["regrets"] /= n_students
        avg_metrics[method_name]["real_rewards"] /= n_students
        avg_metrics[method_name]["cumulated_expected_rewards"] /= n_students
        avg_metrics[method_name]["cumulated_regrets"] /= n_students
        avg_metrics[method_name]["cumulated_real_rewards"] /= n_students

    # Plot results
    rounds = range(n_rounds)
    plt.figure(figsize=(16, 12))

    for method_name in reward_methods.keys():
        plt.subplot(3, 1, 1)
        plt.plot(rounds, avg_metrics[method_name]["expected_rewards"], label=f"({method_name})")
        plt.xlabel("Rounds")
        plt.ylabel("Average Expected Reward")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.subplot(3, 1, 2)
        plt.plot(rounds, avg_metrics[method_name]["regrets"], label=f"({method_name})")
        plt.xlabel("Rounds")
        plt.ylabel("Average Regret")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.subplot(3, 1, 3)
        plt.plot(rounds, avg_metrics[method_name]["real_rewards"], label=f"({method_name})")
        plt.xlabel("Rounds")
        plt.ylabel("Average Real Reward")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    figures_dir = "fig2"
    os.makedirs(figures_dir, exist_ok=True)

    plot_filename_not_cumulated = f"method_comparaison_not_cumulated_dim_{dim_theta}_items_{corpus.nb_items}_rounds_{n_rounds}_expl_{exploration_parameter}_removal_{item_removal}_ke_step_{knowledge_evolving_time_step}_eps_{epsilon}_de_{dynamic_exploration}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, plot_filename_not_cumulated))

    plt.figure(figsize=(16, 8))

    for method_name in reward_methods.keys():
        plt.subplot(3, 1, 1)
        plt.plot(rounds, avg_metrics[method_name]["cumulated_expected_rewards"], label=f"({method_name})")
        plt.xlabel("Rounds")
        plt.ylabel("Average Cumulated Expected Reward")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.subplot(3, 1, 2)
        plt.plot(rounds, avg_metrics[method_name]["cumulated_regrets"], label=f"({method_name})")
        plt.xlabel("Rounds")
        plt.ylabel("Average Cumulated Regret")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.subplot(3, 1, 3)
        plt.plot(rounds, avg_metrics[method_name]["cumulated_real_rewards"], label=f"({method_name})")
        plt.xlabel("Rounds")
        plt.ylabel("Average Cumulated Real Reward")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plot_filename_cumulated = f"mc_cumul_dim_{dim_theta}_items_{corpus.nb_items}_rounds_{n_rounds}_expl_{exploration_parameter}_removal_{item_removal}_ke_step_{knowledge_evolving_time_step}_eps_{epsilon}_de_{dynamic_exploration}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, plot_filename_cumulated))
    
    # Convert avg_metrics to DataFrame
    rows = []
    for method_name in reward_methods.keys():
        for round_num in range(n_rounds):
            rows.append({
                "Method": method_name,
                "Round": round_num,
                "Expected Reward": avg_metrics[method_name]["expected_rewards"][round_num],
                "Regret": avg_metrics[method_name]["regrets"][round_num],
                "Real Reward": avg_metrics[method_name]["real_rewards"][round_num],
                "Cumulated Expected Reward": avg_metrics[method_name]["cumulated_expected_rewards"][round_num],
                "Cumulated Regret": avg_metrics[method_name]["cumulated_regrets"][round_num],
                "Cumulated Real Reward": avg_metrics[method_name]["cumulated_real_rewards"][round_num],
            })

    df_metrics = pd.DataFrame(rows)
    csv_dir = "csv_files"
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = f"avg_metrics_same_students_dim_{dim_theta}_items_{corpus.nb_items}_rounds_{n_rounds}_expl_{exploration_parameter}_removal_{item_removal}_ke_step_{knowledge_evolving_time_step}_eps_{epsilon}_de_{dynamic_exploration}.csv"
    df_metrics.to_csv(os.path.join(csv_dir, csv_filename), index=False)

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the average metrics of n students over n_rounds for different reward methods."
    )
    parser.add_argument("--dim_theta", type=int, default=10, help="Dimension of theta")
    parser.add_argument("--n_students", type=int, default=100, help="Number of students")
    parser.add_argument("--n_rounds", type=int, default=50, help="Number of rounds")
    parser.add_argument("--exploration_parameter", type=float, default=0.1, help="Exploration parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon parameter")
    parser.add_argument("--lambda_", type=float, default=0.01, help="Lambda parameter")
    parser.add_argument("--corpus_nb_items", type=int, default=50, help="Corpus object")
    parser.add_argument("--dynamic_exploration", type=int, default=0, help="Dynamic exploration flag")
    parser.add_argument("--cold_start_len", type=int, default=5, help="Cold start length")
    parser.add_argument("--item_removal", type=int, default=0, help="Item removal flag")
    parser.add_argument("--knowledge_evolving_time_step", type=int, default=0, help="Knowledge evolving time step")

    args = parser.parse_args()
    corpus = Corpus(args.corpus_nb_items, args.dim_theta)

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
        epsilon=args.epsilon,
        dynamic_exploration=args.dynamic_exploration
    )
