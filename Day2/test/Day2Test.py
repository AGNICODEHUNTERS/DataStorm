import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv("../input/UCI_Credit_Card.csv")
data.head(10)

data.columns
