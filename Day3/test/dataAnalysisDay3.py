trainDataSet = pd.read_csv("credit_card_default_train.csv",header=0)
testDataSet = pd.read_csv("credit_card_default_test.csv",header=0)

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
            elif str(mar.iloc[i])=="Other":
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

def normDat(dataSet):
    return (dataSet-train_stats['mean '])/train_stats['std']

def buildModel():
    model = tf.keras.Sequential([tf.keras.layers.Dense(20,kernel_initializer="uniform",activation='relu',input_dim=23),tf.keras.layers.Dense(10,kernel_initializer="uniform",activation="sigmoid"),tf.keras.layers.Dense(5),tf.keras.layers.Dense(3),tf.keras.layers.Dense(1)])
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
    return model

trainDataSet=numeric(trainDataSet)
testDataSet=numeric(testDataSet)


trainDataSet=trainDataSet.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)
testDataSet=trainDataSet.drop(["Client_ID","Balance_Limit_V1","Gender","EDUCATION_STATUS","MARITAL_STATUS","AGE"],axis=1)


X_train=trainDataSet.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
Y_train = trainDataSet["NEXT_MONTH_DEFAULT"].values

X_test = trainDataSet.drop(["NEXT_MONTH_DEFAULT"],axis=1).values
Y_test = trainDataSet["NEXT_MONTH_DEFAULT"].values
