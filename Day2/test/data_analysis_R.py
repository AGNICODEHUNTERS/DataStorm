import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


data = pd.read_csv("credit_card_default_train.csv")

#testData=pd.read_csv("credit_card_default_test")

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

    while True:
        try:
            if str(gen.iloc[i])=="M":
                genF.append(1)
            else:
                genF.append(2)

            if str(edu.iloc[i])=="Graduate":
                eduF.append(1)
            elif str(edu.iloc[i])=="High School":
                eduF.append(2)
            else:
                eduF.append(3)

            if str(mar.iloc[i])=="Single":
                marF.append(1)
            elif str(mar.iloc[i])=="Married":
                marF.append(2)
            else:
                marF.append(3)

            if str(age.iloc[i])=="31-45":
                ageF.append(1)
            elif str(age.iloc[i])=="46-65":
                ageF.append(2)
            else:
                ageF.append(3)
            i=i+1
        except:
            dataSheet.insert(1,"balF",pd.DataFrame(balF))
            dataSheet.insert(2,"genF",pd.DataFrame(genF))
            dataSheet.insert(3,"eduF",pd.DataFrame(eduF))
            dataSheet.insert(4,"marF",pd.DataFrame(marF))
            dataSheet.insert(5,"ageF",pd.DataFrame(ageF))
            break
    return dataSheet

data=numeric(data)
data=data.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
cols = [ f for f in data.columns if data.dtypes[ f ] != "object"]
cols.remove('NEXT_MONTH_DEFAULT')
print(data)
f = pd.melt( data, id_vars='NEXT_MONTH_DEFAULT',  value_vars=cols)
g = sns.FacetGrid( f, hue='NEXT_MONTH_DEFAULT', col="variable", col_wrap=5, sharex=False, sharey=False )
g = g.map( sns.distplot, "value", kde=True).add_legend()


def CSTOW ( inputdata, inputvariable, OutcomeCategory ):
    OutcomeCategoryTable = inputdata.groupby( OutcomeCategory )[ OutcomeCategory ].count().values
    OutcomeCategoryRatio = OutcomeCategoryTable / sum( OutcomeCategoryTable )
    possibleValues = inputdata[inputvariable].unique()
    observed = []
    expected = []
    print(OutcomeCategoryTable)
    print(OutcomeCategoryRatio)
    print(possibleValues)


    for possible in possibleValues:
        countsInCategories = inputdata[ inputdata[ inputvariable ] == possible ].groupby( OutcomeCategory )[OutcomeCategory].count().values
        if( len(countsInCategories) != len( OutcomeCategoryRatio ) ):
            print("Error! The class " + str( possible) +" of \'" + inputvariable + "\' does not contain all values of \'" + OutcomeCategory + "\'" )
            return
        elif( min(countsInCategories) < 5 ):
            print("Chi Squared Test needs at least 5 observations in each cell!")
            print( inputvariable + "=" + str(possible) + " has insufficient data")
            print( countsInCategories )
            return
        else:
            observed.append( countsInCategories )
            expected.append( OutcomeCategoryRatio * len( inputdata[inputdata[ inputvariable ] == possible ]))
    observed = np.array( observed )
    expected = np.array( expected )

    chi_squared_stat = ((observed - expected)**2 / expected).sum().sum()
    degOfF = (observed.shape[0] - 1 ) *(observed.shape[1] - 1 )
    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df=degOfF)
    print("Calculated test-statistic is %.2f" % chi_squared_stat )
    print("If " + OutcomeCategory + " is indep of " + inputvariable + ", this has prob %.2e of occurring" % p_value )
CSTOW(data,'eduF','NEXT_MONTH_DEFAULT')
# The quantitative vars:
quant = ["balF", "ageF"]
# The qualitative but "Encoded" variables (ie most of them)
qual_Enc = cols
qual_Enc.remove("balF")
qual_Enc.remove("ageF")
print(cols)
logged = []
for m in ("PAID_AMT_JULY","PAID_AMT_AUG","PAID_AMT_SEP","PAID_AMT_OCT","PAID_AMT_NOV",'PAID_AMT_DEC'):
    qual_Enc.remove(m)
    print(qual_Enc)
    data[m]  = data[m].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append(m)

for n in ("DUE_AMT_JULY","DUE_AMT_AUG","DUE_AMT_SEP","DUE_AMT_OCT","DUE_AMT_NOV","DUE_AMT_DEC"):
    qual_Enc.remove(n)
    print(qual_Enc)
    data[ n] = data[n].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append(n)

f = pd.melt( data, id_vars='NEXT_MONTH_DEFAULT', value_vars=logged)
g = sns.FacetGrid( f, hue='NEXT_MONTH_DEFAULT', col="variable", col_wrap=3, sharex=False, sharey=False )
g = g.map( sns.distplot, "value", kde=True).add_legend()

features = quant + qual_Enc + logged + ['NEXT_MONTH_DEFAULT']
corr = data[features].corr()
plt.subplots(figsize=(30,10))
sns.heatmap( corr, square=True, annot=True, fmt=".1f" )
plt.show()
