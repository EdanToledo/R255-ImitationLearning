import pandas as pd 
import matplotlib.pyplot as plt

def read_csv(filepath):

    df = pd.read_csv(filepath)

    return df


filename = "/home/edan/Desktop/ImitationLearning/project/plotting/20230302-145604/logs/learner/logs.csv"
df = read_csv(filename)

plt.plot(df["total_loss"])
plt.savefig("temp")