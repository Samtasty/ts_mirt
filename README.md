# ts_mirt

This repository contains different Python scripts for running experiments.

## Main Scripts

The two primary scripts to run are:

- **`benchmark_parallele_same_students.py`**
- **`gr_search_hyperparameter_parallelized.py`**

These scripts are **parallelized**, making them significantly faster.

### `benchmark_parallele_same_students.py`
This script evaluates all methods on the **same students** with the **same initial** knowledge parameter \( \theta \).  
To ensure better empirical guarantees, the argument `--n_students` specifies the number of students tested.

The output consists of **averaged results** over all students for each metric:  
- **Expected Reward**
- **Regret**
- **Real Reward**  
along with their **cumulative counterparts** for each method.  

While each method is tested on the same group of students, individual students within the group have **different initial** \( \theta \) values. Averaging results helps provide a **general trend** of how each method performs.

## Methods Overview

The methods available in `gr_search_hyperparameter_parallelized.py` are also the ones compared in `benchmark_parallele_same_students.py`. Below is a quick reference for the `--method` argument:

- **`expected_reward`**  
  - Uses the **true expected reward** to select the best action in expectation.  
  - Since it has full knowledge, **regret is always zero**.

- **`expected_reward_mle`**  
  - Selects the best action based on the **expected reward computed with** \( \theta_t^{\text{MLE}} \).  
  - No exploration is applied.

- **`epsilon_greedy`**  
  - Classical **epsilon-greedy** strategy.  
  - With probability \( \epsilon \), a random item is selected; otherwise, the best action given \( \theta_t^{\text{MLE}} \) is chosen.

- **`expected_reward_ucb`**  
  - Implements the **GLM-UCB** algorithm.  
  - The argument `--exploration_parameter` controls the exploration-exploitation balance.

- **`expected_reward_ellipsoid`**  
  - A variation of `expected_reward_ucb`, applying **Faury's algorithm**.  
  - Improves the confidence interval by considering the **local curvature of the sigmoid function**.

- **`expected_reward_fisher`**  
  - **Fisher Information-based exploration method**.  
  - Instead of using the design matrix, this method favors actions that **maximize Fisher information** to gain more knowledge about \( \theta \).  
  - The argument `--exploration_parameter` adjusts the exploration-exploitation balance.

- **`expected_reward_ts`**  
  - Implements **classical Thompson Sampling**.  
  - Samples \( \theta_t^{\text{TS}} \) from the **posterior distribution**, computed using **Laplace Approximation**.  
  - Selects the best action based on the sampled \( \theta_t^{\text{TS}} \).

- **`expected_reward_proxi`**  
  - Uses **Laplace approximation** and the specific shape of the reward function to directly infer the **expected outcome given an action**.  
  - Instead of sampling \( \theta \), it integrates over the posterior to compute the expected reward and selects the best action.

## Other Arguments

Below is a summary of additional arguments that can be used:

| Argument | Description | Default Value |
|----------|------------|--------------|
| `--dim_theta` | Dimension of \( \theta \) (number of knowledge components defining the corpus). | *Required* |
| `--n_students` | Number of students to test the methods on (for better empirical guarantees). | `100` |
| `--n_rounds` | Number of items given to students after the cold start. | `50` |
| `--exploration_parameter` | Exploration-exploitation trade-off parameter. | `0.1` |
| `--epsilon` | Epsilon parameter for the epsilon-greedy method. | `0.1` |
| `--lambda_` | Regularization parameter ensuring invertibility of the Fisher Information Matrix or design matrix. | `0.01` |
| `--corpus_nb_items` | Number of items in the corpus (must be larger than `n_rounds` if `item_removal=True`). | `50` |
| `--cold_start_len` | Length of the cold start phase (random items given to students before learning begins). | `5` |
| `--item_removal` | Whether to remove items from the available corpus after selection. | `False` |
| `--knowledge_evolving_time_step` | Number of rounds after which student knowledge evolves. If `0`, knowledge remains fixed. | `0` |

---

