from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc, rcParams
from matplotlib.ticker import MaxNLocator
from rliable import library as rly
from rliable import metrics, plot_utils

sns.set_style("white")

rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

rc("text", usetex=False)

# Huge credit to:
# https://colab.research.google.com/drive/1a0pSD-1tWhMmeJeeoyZM1A-HCW3yf1xR?usp=sharing#scrollTo=G9OXgSZTPBoQ
# for library and code to create a lot of these plots. I have simply wrapped their functions to be more
# self contained and user friendly as well as added docstrings.


def score_normalization(score: float, min_score: float, max_score: float) -> float:
    """Normalises score.

    Args:
        score (float): score to normalise.
        min_score (float): minimum score achievable.
        max_score (float): maximimum score achievable.

    Returns:
        float: normalised score.
    """
    return (score - min_score) / (max_score - min_score)


def aggregate_metrics(data_dict: Dict[str, np.ndarray], optimality_threshold=1):
    """Generate plots for aggregate metrics with 95% Stratified Bootstrap CIs

    'Median', 'IQM', 'Mean', and 'Optimality Gap'

    Args:
        data_dict (Dict[str, np.ndarray]): Dictionary mapping algorithm to data (num_runs x num_envs)
        optimality_threshold (int) : score threshold deemed optimal (or desired performance score) - depends on how data is normalised but generally should be 1.

    Returns:
        fig: A matplotlib Figure.
        axes: `axes.Axes` or array of Axes.

    """

    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x, optimality_threshold),
        ]
    )
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        data_dict, aggregate_func, reps=50000
    )
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        xlabel="Normalized Score",
    )

    return fig, axes


def probability_of_improvement(algorithm_pairs: Dict[str, Tuple[float, float]]):
    """Generate plot of probability of improvement when comparing algorithms.

    Args:
        algorithm_pairs (Dict[str, Tuple[float,float]]): Dictionary containing pairs of algorithms to compare mapping to pairs of scores. e.g. {"algo_x,algo_y" : (100, 200) }

    Returns:
         `axes.Axes` : which contains the plot for probability of improvement.
    """

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        algorithm_pairs, metrics.probability_of_improvement, reps=2000
    )

    return plot_utils.plot_probability_of_improvement(
        average_probabilities, average_prob_cis
    )


def sample_efficiency_curve(data_dict: Dict[str, np.ndarray], interval: int):
    """Generate sample efficiency plots.

    Args:
        data_dict (Dict[str, np.ndarray]): dictionary mapping algorithms to their normalized score matrices across all training timesteps,
            each of which is of size `(num_runs x num_games x num_timesteps)`.
        interval (int) : the interval at which to plot the scores.

    Returns:
        `axes.Axes` : object containing the plot.
    """

    timesteps = np.arange(1, list(data_dict.values())[0].shape[-1], interval) - 1
    data_timesteps_scores_dict = {
        algorithm: score[:, :, timesteps] for algorithm, score in data_dict.items()
    }
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])]
    )
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        data_timesteps_scores_dict, iqm, reps=50000
    )
    return plot_utils.plot_sample_efficiency_curve(
        timesteps + 1,
        iqm_scores,
        iqm_cis,
        xlabel=r"Number of Timesteps",
        ylabel="IQM Normalized Score",
        algorithms=None,
    )


def performance_profiles(
    data_dict: Dict[str, np.ndarray], normalised_score_threshold=1.0
):
    """Generate plots for performance profiles.

    Args:
        data_dict (Dict[str, np.ndarray]): Dictionary mapping algorithm to data (num_runs x num_envs)

    Returns:
        fig: A matplotlib Figure.
        axes: `axes.Axes` or array of Axes.

    """
    algorithms = list(data_dict.keys())
    # normalized score thresholds
    normalised_score_thresholds = np.linspace(0.0, normalised_score_threshold, 81)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        data_dict, normalised_score_thresholds
    )
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    return plot_utils.plot_performance_profiles(
        score_distributions,
        normalised_score_thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(algorithms, sns.color_palette("colorblind"))),
        xlabel=r"Normalized Score $(\tau)$",
        ax=ax,
    )


def decorate_axis(ax, wrect=10, hrect=10, labelsize="large"):
    """Plotting utility function"""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))


def plot_score_hist(
    score_matrix,
    names,
    bins=20,
    figsize=(28, 14),
    fontsize="xx-large",
    N=6,
    extra_row=1,
):
    """Plot histograms of performance of runs.

    Args:
        score_matrix (np.ndarray): matrix containing scores on environments (num_runs, num_envs)
        names (list(str)): The list of names of the environments whose scores are in score_matrix.
        bins (int, optional): The number of bins in the histogram. Defaults to 20.
        figsize (tuple, optional): The size of the figure. Defaults to (28, 14).
        fontsize (str, optional): The size of the font on the figure. Defaults to 'xx-large'.
        N (int, optional): The number of columns in plot. Defaults to 6.
        extra_row (int, optional): The number of extra rows. Defaults to 1.


    Returns:
        fig: A matplotlib Figure.
    """
    num_tasks = score_matrix.shape[1]

    N1 = (num_tasks // N) + extra_row
    fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
    for i in range(N):
        for j in range(N1):
            idx = j * N + i
            if idx < num_tasks:
                ax[j, i].set_title(names[idx], fontsize=fontsize)
                sns.histplot(score_matrix[:, idx], bins=bins, ax=ax[j, i], kde=True)
            else:
                ax[j, i].axis("off")
            decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize="xx-large")
            ax[j, i].xaxis.set_major_locator(MaxNLocator(4))
            if idx % N == 0:
                ax[j, i].set_ylabel("Count", size=fontsize)
            else:
                ax[j, i].yaxis.label.set_visible(False)
            ax[j, i].grid(axis="y", alpha=0.1)
    return fig