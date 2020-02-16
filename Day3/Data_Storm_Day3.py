import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

dataSetT = pd.read_csv("credit_card_default_train.csv",header=0)
dataSetTs = pd.read_csv("credit_card_default_test.csv",header=0)

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

def normDat(dataSet):
    return (dataSet-train_stats['mean'])/train_stats['std']

def buildModel():
    model = tf.keras.Sequential([tf.keras.layers.Dense(20,kernel_initializer="uniform",activation='relu',input_dim=23),tf.keras.layers.Dense(10,kernel_initializer="uniform",activation="sigmoid"),tf.keras.layers.Dense(5),tf.keras.layers.Dense(3),tf.keras.layers.Dense(1)])
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
    return model

data=numeric(dataSetT)
testData=numeric(dataSetTs)


train_dat=data.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
test_dat=testData.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)


train_dat=train_dat.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
train_lab = data["NEXT_MONTH_DEFAULT"].values

'''train_dat.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
train_lab = train_dat["NEXT_MONTH_DEFAULT"].values'''

sc=StandardScaler()
train_dat = sc.fit_transform(train_dat)
test_dat = sc.transform(test_dat)

print(train_dat)
print(test_dat)

model = buildModel()

model.fit(train_dat,train_lab,batch_size=10,epochs=500)
test_predictions = model.predict(test_dat).flatten()

print(test_predictions)
tp=[]
for i in test_predictions:
    fin=int(round(i))
    if fin>1:
        fin=1
    if fin<0:
        fin=0
    tp.append(fin)
dataSheet=pd.DataFrame()
dataSheet.insert(0,"Client_ID",testData.Client_ID)
dataSheet.insert(1,"NEXT_MONTH_DEFAULT",pd.DataFrame(tp))
dataSheet.to_csv(r'AGNI_CODE_HUNTERS.csv')
 
