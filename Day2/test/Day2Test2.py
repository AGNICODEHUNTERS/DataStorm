import tensorflow as tf
import pandas as pd
import seaborn as sns
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt

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
    model = tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu',input_shape=[len(train_dat.keys())]),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    return model

data=numeric(dataSetT)
testData=numeric(dataSetTs)


train_dat=data.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
test_dat=testData.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)


train_lab = train_dat.pop("NEXT_MONTH_DEFAULT")
test_lab = test_dat.pop("NEXT_MONTH_DEFAULT")

train_stats=train_dat.describe()
train_stats = train_stats.transpose()
normed_train_dat = normDat(train_dat)
normed_test_dat = normDat(test_dat)
model = buildModel()

EPOCHS=1000
history = model.fit(normed_train_dat,train_lab,epochs = EPOCHS,validation_split = 0.2,verbose = 0,callbacks=[tfdocs.modeling.EpochDots()])
loss, mae, mse = model.evaluate(normed_test_dat, test_lab, verbose=2)
test_predictions = model.predict(normed_test_dat).flatten()
tp=[]
for i in test_predictions:
    fin=int(round(i))
    tp.append(abs(fin))
dataSheet=pd.DataFrame()
dataSheet.insert(0,"Client_ID",data.Client_ID)
dataSheet.insert(1,"NEXT_MONTH_DEFAULT",pd.DataFrame(tp))
dataSheet.to_csv(r'AGNI_CODE_HUNTERS.csv')
print(count/len(train_lab))
