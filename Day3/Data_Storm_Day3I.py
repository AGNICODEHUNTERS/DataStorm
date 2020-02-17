import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

trainDataSet = pd.read_csv("credit_card_default_train.csv",header=0)
testDataSet = pd.read_csv("credit_card_default_test.csv",header=0)
testClIds = testDataSet.Client_ID

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
    model = tf.keras.Sequential([tf.keras.layers.Dense(2,kernel_initializer="linear",activation='relu',input_dim=32)])
    #model.add(tf.keras.layers.Dense(32,kernel_initializer="uniform",activation="sigmoid"))
    #model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Dense(8,activation="elu"))
    '''model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Dense(2))'''
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
    return model

trainDataSet=pd.get_dummies(trainDataSet,columns=["Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],drop_first=True).drop(["Client_ID"],axis=1)
testDataSet=pd.get_dummies(testDataSet,columns=["Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],drop_first=True).drop(["Client_ID"],axis=1)

#trainDataSet=trainDataSet.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
#testDatSet=testDataSet.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)

X_train = trainDataSet.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
Y_train = trainDataSet["NEXT_MONTH_DEFAULT"].values
X_test = testDataSet.values
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
sc=StandardScaler()
mm=MinMaxScaler()
nn=Normalizer()
qt=QuantileTransformer()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)


model = buildModel()

model.fit(X_train,Y_train,batch_size=10,epochs=1000)
test_predictions = model.predict(X_test).flatten()

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
dataSheet.insert(0,"Client_ID",testClIds)
dataSheet.insert(1,"NEXT_MONTH_DEFAULT",pd.DataFrame(tp))
dataSheet.to_csv(r'AGNI_CODE_HUNTERSI.csv')

'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, tp)
print(cm)'''
