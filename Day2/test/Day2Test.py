import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv("credit_card_default_test.csv")
print(data.head(10))

print(data.columns)
