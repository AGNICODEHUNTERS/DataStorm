import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv("credit_card_default_train.csv")
output='NEXT_MONTH_DEFAULT'
cols = [ f for f in data.columns if data.dtypes[ f ] != "object"]
#cols.remove('Client_ID')
cols.remove( output )
f = pd.melt( data, id_vars=output, value_vars=cols)
print(f)
g = sns.FacetGrid( f, hue=output, col="variable", col_wrap=5, sharex=False, sharey=False )
g = g.map( sns.distplot, "value", kde=True).add_legend()
plt.show()

''''
def numeric(dataSheet):
    bal = dataSheet.Balance_Limit_V1
    gen = dataSheet.Gender
    edu = dataSheet.EDUCATION_STATUS
    mar = dataSheet.MARITAL_STATUS
    age = dataSheet.AGE

    i=0

    balF = []
    genF = []
    eduF = []
    marF = []
    ageF = []

    for m in range(len(bal)):
        if bal.iloc[m].endswith('M'):
            balF.append(float(bal[m][:-1])*1000000)
        elif bal.iloc[m].endswith('K'):
            balF.append(float(bal[m][:-1])*1000)
        else:
            balF.append(float(bal[m]))
    return dataSheet
data=numeric(data)
print(data)'''
