import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

data = pd.read_csv("credit_card_default_train.csv",header=0)
testData = pd.read_csv("credit_card_default_test.csv",header=0)


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
testData=numeric(testData)
print(data)

y_train=data.NEXT_MONTH_DEFAULT
x_train=data.drop(["NEXT_MONTH_DEFAULT","Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
x_test=testData.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
y_test=[]

from sklearn.linear_model import LinearRegression as lm

model=lm().fit(x_train,y_train)
predictions=model.predict(x_test)
predictionR=[]
for i in predictions:
    predictionR.append(int(round(i)))
datasheet = pd.DataFrame()
datasheet.insert(0,"Client_ID",testData.Client_ID)
datasheet.insert(1,"NEXT_MONTH_DEFAULT",pd.DataFrame(predictionR))
datasheet.to_csv(r'file.csv')

plt.show()
