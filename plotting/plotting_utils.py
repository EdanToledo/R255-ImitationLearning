from collections import defaultdict
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


normalisation_values = {"HalfCheetah" : (-280.18, 12135.0), "hopper" : (-20.27, 3234.3), "Walker2d" : (1.63, 4592.3)}

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def read_all_csvs(csv_folder_path, only_imitation_learning = False):
    files = [f for f in listdir(csv_folder_path) if isfile(join(csv_folder_path, f)) and f != ".DS_Store"]
    files.sort(key=natural_keys)

    data = defaultdict(lambda : defaultdict(lambda : {}))
    for filename in files:
        filepath = join(csv_folder_path, filename)
        
        vals = filename.split("_")
        alg = vals[0][4:]
        seed = vals[2]
        env = vals[-2][:vals[-2].index("-")]
        if only_imitation_learning:
            if vals[3] == "dataset":
                datasize = vals[4]
                if datasize == "None":
                    datasize = "100%"
                else:
                    datasize = str(int(datasize)//10) + "%"
                alg = alg+" "+datasize
            else:
                continue
        

        df = pd.read_csv(filepath)
        run_evaluation_scores = np.array(df["Value"])

        run_evaluation_scores = score_normalisation(run_evaluation_scores, *normalisation_values[env])

        run_evaluation_scores = run_evaluation_scores.reshape(1,1,-1)

        data[alg][seed][env] = run_evaluation_scores
       
    final_data = {}
    
    for alg, d in data.items():
        alg_scores = None
        for seed, di in d.items():
            env_scores = None
            for env, score in di.items():
                
                if env_scores is None:
                    env_scores = score
                else:
                    env_scores = np.concatenate([env_scores, score],1)
            
            if alg_scores is None:
                alg_scores = env_scores
            else:
                alg_scores = np.concatenate([alg_scores, env_scores], 0)
 
        
        final_data[alg] = alg_scores


    return final_data

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
        

def plot_scores(folder_path):
    evaluation_metrics = read_all_csvs(folder_path)
    
    sorted_keys = sorted(evaluation_metrics.keys(), key=natural_keys)

    sorted_eval_metrics = {key : evaluation_metrics[key] for key in sorted_keys}
    
    ax = sample_efficiency_curve(sorted_eval_metrics, 40, 500)
    plt.legend()

    sorted_eval_metrics = calculate_final_scores(sorted_eval_metrics)

    fig, axes = aggregate_metrics(sorted_eval_metrics)

    plt.show()

if __name__ == "__main__":
    folder = "/Users/edantoledo/UniversityWork/ImitationLearning/csvs"
    
    plot_scores(folder)