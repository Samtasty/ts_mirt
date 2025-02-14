import os
import sys
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# For parallelization
from joblib import Parallel, delayed

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


def run_one_student(
    dim_theta,
    method_name,
    n_rounds,
    exploration_parameter,
    lambda_,
    corpus,
    cold_start_len,
    item_removal,
    knowledge_evolving_time_step,
    epsilon,
):
    """
    Simulate one student using the specified bandit method and hyperparams.
    Return arrays: expected_rewards, regrets, real_rewards (each of length n_rounds).
    """
    # Generate one student
    gen_student = generate_student(dim_theta)
    theta = gen_student.theta

    # Initialize BayesianStudent (or whichever Student class)
    student = BayesianStudent(
        theta,
        corpus=corpus,
        mu=np.zeros(gen_student.theta_dim),
        sigma=np.eye(gen_student.theta_dim),
        lambda_=lambda_,
    )

    # Cold start
    cold_start_items = np.random.randint(0, corpus.nb_items, size=cold_start_len)
    student.simulate_pluriel(corpus.list_items(cold_start_items))

    # If knowledge evolves in chunks
    if knowledge_evolving_time_step > 0:
        chunks = n_rounds // knowledge_evolving_time_step
        for _ in range(chunks):
            student.bandit_simulation(
                knowledge_evolving_time_step,
                exploration_parameter=exploration_parameter,
                reward_method_name=method_name,
                epsilon=epsilon,
                item_removal=item_removal,
            )
            # then artificially "improve" knowledge
            student.improvement(generate_learning_gains(gen_student.theta_dim))
        # leftover steps if n_rounds not divisible by knowledge_evolving_time_step
        leftover = n_rounds % knowledge_evolving_time_step
        if leftover > 0:
            student.bandit_simulation(
                leftover,
                exploration_parameter=exploration_parameter,
                reward_method_name=method_name,
                epsilon=epsilon,
                item_removal=item_removal,
            )
    else:
        # Single chunk
        student.bandit_simulation(
            n_rounds,
            exploration_parameter=exploration_parameter,
            reward_method_name=method_name,
            epsilon=epsilon,
            item_removal=item_removal,
        )

    # Return the arrays of interest
    return (
        np.array(student.expected_reward_list),
        np.array(student.regrets_list),
        np.array(student.rewards_list),
        np.cumsum(student.expected_reward_list),
        np.cumsum(student.regrets_list),
        np.cumsum(student.rewards_list),
    )


def run_experiment_parallel(
    method_name,
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
):
    """
    Parallel version of run_experiment:
    - Spawns multiple processes for each student using joblib.
    - Returns a DataFrame with columns: [Round, Expected Reward, Regret, Real Reward].
    """

    # 1) Parallelize over n_students
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_student)(
            dim_theta=dim_theta,
            method_name=method_name,
            n_rounds=n_rounds,
            exploration_parameter=exploration_parameter,
            lambda_=lambda_,
            corpus=corpus,
            cold_start_len=cold_start_len,
            item_removal=item_removal,
            knowledge_evolving_time_step=knowledge_evolving_time_step,
            epsilon=epsilon,
        )
        for _ in range(n_students)
    )

    # 2) Each entry in 'results' is a tuple (exp_rewards, regrets, real_rewards)
    sum_expected = np.zeros(n_rounds)
    sum_regrets = np.zeros(n_rounds)
    sum_real_rewards = np.zeros(n_rounds)
    cum_sum_expected = np.zeros(n_rounds)
    cum_sum_regrets = np.zeros(n_rounds)
    cum_sum_real_rewards = np.zeros(n_rounds)

    for exp_r, rg, real_r, cum_exp_r, cum_rg, cum_real_r in results:
        sum_expected += exp_r
        sum_regrets += rg
        sum_real_rewards += real_r
        cum_sum_expected += cum_exp_r
        cum_sum_regrets += cum_rg
        cum_sum_real_rewards += cum_real_r

    # 3) Average over students
    avg_expected_rewards = sum_expected / n_students
    avg_regrets = sum_regrets / n_students
    avg_real_rewards = sum_real_rewards / n_students

    avg_cum_expected_rewards = cum_sum_expected / n_students
    avg_cum_regrets = cum_sum_regrets / n_students
    avg_cum_real_rewards = cum_sum_real_rewards / n_students

    # 4) Build and return a DataFrame
    df_results = pd.DataFrame(
        {
            "Round": np.arange(n_rounds),
            "Expected Reward": avg_expected_rewards,
            "Regret": avg_regrets,
            "Real Reward": avg_real_rewards,
            "Cumulative Expected Reward": avg_cum_expected_rewards,
            "Cumulative Regret": avg_cum_regrets,
            "Cumulative Real Reward": avg_cum_real_rewards,
        }
    )
    return df_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot the average metrics for a single reward method over n_rounds, \
                     across multiple exploration_parameter values (grid search)."
    )
    parser.add_argument("--dim_theta", type=int, default=10, help="Dimension of theta")
    parser.add_argument(
        "--n_students", type=int, default=100, help="Number of students"
    )
    parser.add_argument("--n_rounds", type=int, default=50, help="Number of rounds")
    parser.add_argument(
        "--exploration_parameter",
        type=float,
        default=0.1,
        help="(Deprecated) Single exploration parameter - not used directly, we do grid search now.",
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon parameter")
    parser.add_argument("--lambda_", type=float, default=0.01, help="Lambda parameter")
    parser.add_argument("--corpus_nb_items", type=int, default=50, help="Corpus object")
    parser.add_argument(
        "--cold_start_len", type=int, default=5, help="Cold start length"
    )
    parser.add_argument(
        "--item_removal", type=bool, default=False, help="Item removal flag"
    )
    parser.add_argument(
        "--knowledge_evolving_time_step",
        type=int,
        default=0,
        help="Knowledge evolving time step",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="expected_reward",
        help='Which single reward method to run. E.g. "expected_reward", "epsilon_greedy", etc.',
    )

    args = parser.parse_args()

    # Construct the corpus
    corpus = Corpus(args.corpus_nb_items, args.dim_theta)

    # Define a grid of exploration parameters to try
    exploration_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    all_results = []

    for epar in exploration_grid:
        print(f"\nRunning method='{args.method}' with exploration_parameter={epar} ...")
        df_metrics = run_experiment_parallel(
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
            epsilon=args.epsilon,
        )
        # Tag each row with the exploration_parameter used
        df_metrics["exploration_parameter"] = epar
        all_results.append(df_metrics)

    # Concatenate all results
    df_all = pd.concat(all_results, ignore_index=True)

    csv_dir = "csv_files"
    os.makedirs(csv_dir, exist_ok=True)

    outname_csv = os.path.join(
        csv_dir,
        f"gs_ir_{args.item_removal}_nc_{args.corpus_nb_items}_dim_{args.dim_theta}_ke_{args.knowledge_evolving_time_step}s{args.method}_results.csv",
    )
    df_all.to_csv(outname_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 8))
    metrics = ["Cumulative Expected Reward", "Cumulative Regret", "Cumulative Real Reward"]

    for idx, metric in enumerate(metrics, start=1):
        plt.subplot(3, 1, idx)
        for epar in exploration_grid:
            subset = df_all[df_all["exploration_parameter"] == epar]
            plt.plot(subset["Round"], subset[metric], label=f"{metric}, expl={epar}")
        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.legend()

    plt.suptitle(f"Method: {args.method} | Grid Search over exploration_parameter")
    plt.tight_layout()
    # Ensure the 'figures' directory exists
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Save the figure to the 'figures' folder
    outname_pdf = os.path.join(
        figures_dir,
        f"gs_ir_{args.item_removal}_nc_{args.corpus_nb_items}_dim_{args.dim_theta}_ke_{args.knowledge_evolving_time_step}s{args.method}_plot.pdf",
    )
    plt.savefig(outname_pdf)
    outname_pdf = f"parallele_grid_search_{args.method}_plot.pdf"
    plt.savefig(outname_pdf)

    print(
        f"\nDone! The results are in '{outname_csv}' and the plot is in '{outname_pdf}'."
    )
