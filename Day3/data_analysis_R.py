import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


data = pd.read_csv("credit_card_default_train.csv")

testData=pd.read_csv("credit_card_default_test.csv")

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

            else:
                marF.append(2)

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
testData=numeric(testData)
testData=testData.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
cols = [ f for f in data.columns if data.dtypes[ f ] != "object"]
colstest = [ f for f in testData.columns if testData.dtypes[ f ] != "object"]
cols.remove('NEXT_MONTH_DEFAULT')
for i in cols:
    f = pd.melt( data, id_vars='NEXT_MONTH_DEFAULT',  value_vars=i)
    g = sns.FacetGrid( f, hue='NEXT_MONTH_DEFAULT', col="variable", col_wrap=1, sharex=False, sharey=False )
    g = g.map( sns.distplot, "value", kde=True).add_legend()
    #plt.savefig(i+".png")


def CSTOW ( inputdata, inputvariable, OutcomeCategory ):
    OutcomeCategoryTable = inputdata.groupby( OutcomeCategory )[ OutcomeCategory ].count().values
    OutcomeCategoryRatio = OutcomeCategoryTable / sum( OutcomeCategoryTable )
    possibleValues = inputdata[inputvariable].unique()
    observed = []
    expected = []

    for possible in possibleValues:
        countsInCategories = inputdata[ inputdata[ inputvariable ] == possible ].groupby( OutcomeCategory )[OutcomeCategory].count().values
        if( len(countsInCategories) != len( OutcomeCategoryRatio ) ):
            print("Error! The class " + str( possible) +" of \'" + inputvariable + "\' does not contain all values of \'" + OutcomeCategory + "\'" )
            return
        elif( min(countsInCategories) < 5 ):
            print("Chi Squared Test needs at least 5 observations in each cell!")
            print( inputvariable + "=" + str(possible) + " has insufficient data")

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

# The quantitative vars:
quant = ["balF", "ageF"]
# The qualitative but "Encoded" variables (ie most of them)
qual_Enc = cols
qual_Enc.remove("balF")
qual_Enc.remove("ageF")
logged = []
for m in ("PAID_AMT_JULY","PAID_AMT_AUG","PAID_AMT_SEP","PAID_AMT_OCT","PAID_AMT_NOV",'PAID_AMT_DEC'):
    qual_Enc.remove(m)
    data[m]  = data[m].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append(m)

for n in ("DUE_AMT_JULY","DUE_AMT_AUG","DUE_AMT_SEP","DUE_AMT_OCT","DUE_AMT_NOV","DUE_AMT_DEC"):
    qual_Enc.remove(n)
    data[ n] = data[n].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append(n)
###############################
# The quantitative vars:
quant = ["balF", "ageF"]
# The qualitative but "Encoded" variables (ie most of them)
qual_Enctest = colstest
qual_Enctest.remove("balF")
qual_Enctest.remove("ageF")
loggedtest = []
for m in ("PAID_AMT_JULY","PAID_AMT_AUG","PAID_AMT_SEP","PAID_AMT_OCT","PAID_AMT_NOV",'PAID_AMT_DEC'):
    qual_Enctest.remove(m)
    testData[m]  = testData[m].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    loggedtest.append(m)

for n in ("DUE_AMT_JULY","DUE_AMT_AUG","DUE_AMT_SEP","DUE_AMT_OCT","DUE_AMT_NOV","DUE_AMT_DEC"):
    qual_Enctest.remove(n)
    testData[ n] = testData[n].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    loggedtest.append(n)

################################
for i in logged:
    f = pd.melt( data, id_vars='NEXT_MONTH_DEFAULT', value_vars=i)
    g = sns.FacetGrid( f, hue='NEXT_MONTH_DEFAULT', col="variable", col_wrap=1, sharex=False, sharey=False )
    g = g.map( sns.distplot, "value", kde=True).add_legend()

features = quant + qual_Enc + logged + ['NEXT_MONTH_DEFAULT']
corr = data[features].corr()
plt.subplots(figsize=(30,10))
sns.heatmap( corr, square=True, annot=True, fmt=".1f" )

'''X_train=data.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
Y_train = data["NEXT_MONTH_DEFAULT"].values

X_test = data.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
Y_test = data["NEXT_MONTH_DEFAULT"].values'''

X= data.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
Y=data["NEXT_MONTH_DEFAULT"].values
featurestest= quant + qual_Enctest + loggedtest
features = quant + qual_Enc + logged
'''X_train= data[features].values
X_test= testData[featurestest].values
Y_train= data[ "NEXT_MONTH_DEFAULT" ].values'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform( X_train )
X_test = scX.transform( X_test )

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit( X_train, Y_train )
Y_pred = classifier.predict( X_test )

cm = confusion_matrix( Y_test, Y_pred )
print("Accuracy on Test Set for RandomForest = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresRF = cross_val_score( classifier, X_train, Y_train, cv=10)
print("Mean RandomForest CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresRF.mean(), scoresRF.std() ))
print(Y_pred)

from sklearn.svm import SVC
classifier1 = SVC(kernel="rbf")
classifier1.fit( X_train, Y_train )
Y_pred = classifier1.predict( X_test )

cm = confusion_matrix( Y_test, Y_pred )
print("Accuracy on Test Set for kernel-SVM = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresSVC = cross_val_score( classifier1, X_train, Y_train, cv=10)
print("Mean kernel-SVM CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresSVC.mean(), scoresSVC.std() ))


from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression()
classifier2.fit( X_train, Y_train )
Y_pred = classifier2.predict( X_test )

cm = confusion_matrix( Y_test, Y_pred )
print("Accuracy on Test Set for LogReg = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresLR = cross_val_score( classifier2, X_train, Y_train, cv=10)
print("Mean LogReg CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresLR.mean(), scoresLR.std() ))

from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit( X_train, Y_train )
Y_pred = classifier3.predict( X_test )
cm = confusion_matrix(Y_test, Y_pred )
print("Accuracy on Test Set for NBClassifier = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresNB = cross_val_score( classifier3, X_train, Y_train, cv=10)
print("Mean NaiveBayes CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresNB.mean(), scoresNB.std() ))


from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors=5)
classifier4.fit( X_train, Y_train )
Y_pred = classifier4.predict( X_test )
cm = confusion_matrix( Y_test, Y_pred )
print("Accuracy on Test Set for KNeighborsClassifier = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresKN = cross_val_score( classifier3, X_train, Y_train, cv=10)
print("Mean KN CrossVal Accuracy on Train Set Set %.2f, with std=%.2f" % (scoresKN.mean(), scoresKN.std() ))

Ran = RandomForestClassifier(criterion= 'gini', max_depth= 6,
                                     max_features= 5, n_estimators= 150,
                                     random_state=0)
Ran.fit(X_train, Y_train)
Y_pred = Ran.predict(X_test)
print('Accuracy:', metrics.accuracy_score(Y_pred,Y_test))

## 5-fold cross-validation
cv_scores =cross_val_score(Ran, X, Y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(Y_test,Y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(y_test,Ran.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Random Forest");












df = pd.DataFrame(Y_pred)
df.to_csv(r'a.csv')
