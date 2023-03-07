import functools
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd 
from stats_utils import sample_efficiency_curve, performance_profiles, score_normalisation, aggregate_metrics
import re 
#@title Plotting: Seaborn style and matplotlib params
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("white")


normalisation_values = {"halfcheetah" : (-280.18, 12135.0), "hopper" : (-20.27, 3234.3), "walker2d" : (1.63, 4592.3)}

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def read_all_csvs(csv_folder_path):
    files = [f for f in listdir(csv_folder_path) if isfile(join(csv_folder_path, f))]
    files.sort(key=natural_keys)

    data = {}
    for filename in files:
        filepath = join(csv_folder_path, filename)
        df = pd.read_csv(filepath)
        run_evaluation_scores = np.array(df["Value"])
        # Temporary - needs to be in shape num runs x num envs x num timesteps
        run_evaluation_scores = np.expand_dims(np.expand_dims(run_evaluation_scores, 0), 0)
        
        data[filename[4:filename.index("_seed")]] = run_evaluation_scores


    return data

def calculate_final_scores(evaluation_metrics):

    for alg in evaluation_metrics:
        evaluation_metrics[alg] =  np.mean(evaluation_metrics[alg][:,:, -100:], axis=-1)

    return evaluation_metrics

def normalise_evaluation_metrics(evaluation_metrics, min_score, max_score):
    for alg in evaluation_metrics:
        evaluation_metrics[alg] =  np.apply_along_axis(functools.partial(score_normalisation, min_score = min_score, max_score = max_score), axis=-1, arr=evaluation_metrics[alg])

    return evaluation_metrics

def add_dummy_data(evaluation_metrics, num_runs):
    for alg in evaluation_metrics:
        for _ in range(num_runs):
            noise = np.random.normal(0,0.1)
            noisy_run = evaluation_metrics[alg] + noise
            evaluation_metrics[alg] =  np.concatenate([evaluation_metrics[alg], noisy_run], axis=0)

    return evaluation_metrics
        
        
       

def plot_scores(folder_path, environment_name):
    evaluation_metrics = read_all_csvs(folder_path)
    evaluation_metrics = normalise_evaluation_metrics(evaluation_metrics, *normalisation_values[environment_name])

    # evaluation_metrics = add_dummy_data(evaluation_metrics, 2)
    
    ax = sample_efficiency_curve(evaluation_metrics, 50)
    plt.legend()
    plt.savefig(f"SampleEfficiency_{environment_name}")

    evaluation_metrics = calculate_final_scores(evaluation_metrics)

    ax = performance_profiles(evaluation_metrics)
    plt.legend()
    plt.savefig(f"PerformanceProfile_{environment_name}")

   
    fig, axes = aggregate_metrics(evaluation_metrics)
    plt.savefig(f"AggregateMetrics_{environment_name}")
    
    plt.show()

if __name__ == "__main__":
    folder = "/Users/edantoledo/UniversityWork/ImitationLearning/csvs/halfcheetah/seed_0"
    
    plot_scores(folder, "halfcheetah")