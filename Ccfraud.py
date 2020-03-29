import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataframe= pd.read_csv("PS_20174392719_1491204439457_log.csv")
del dataframe['nameDest']
del dataframe['nameOrig']
del dataframe['isFlaggedFraud']
dataframe1=dataframe[0:-1]
#a=dataframe['isFraud']
#cnt0=0
#cnt1=1

'''
for i in range(len(a)):
    if a[i]==0:
      cnt0+=1
    else:
        cnt1+=1

fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
name=['zero', 'one']
number=[cnt0,cnt1]
ax.bar(name,number)
plt.show()
'''

le=preprocessing.LabelEncoder()
dataframe1['type2']=le.fit_transform(dataframe1['type'])
del dataframe1['type']
shuffled_df=dataframe1.sample(frac=1,random_state=4)
fraud_df=shuffled_df.loc[shuffled_df['isFraud']==1]
print(len(fraud_df))
non_fraud_df=shuffled_df.loc[shuffled_df['isFraud']==0].sample(8212,random_state=4)
normalized=pd.concat([fraud_df,non_fraud_df])

#a1=normalized['isFraud']
"""cnt2=0
cnt3=0
i=0
"""
'''
for i in range(len(a1)):
    if a1[i]==0:
        cnt2=cnt2+1
    else:
        cnt3=cnt3+1
fig1=plt.figure()
name=["fraud","notfraud"]
number=[cnt2,cnt3]
ax=fig1.add_axes([0,0,1,1])
ax.bar(name,number, align='center')
plt.xlabel("Classes")
plt.ylabel("Number of Cases")
plt.show()
'''

normalized1=normalized.sample(frac=1, random_state=4)

X=normalized1.drop(['isFraud'], axis=1)
Y=normalized1['isFraud']
y3=Y.values.reshape(-1,1)
st_sc = StandardScaler()
X = st_sc.fit_transform(X)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
Y1=ohe.fit_transform(y3).toarray()
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train,y_test=train_test_split(X,Y1, test_size=0.2, random_state=1)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(units = 64, input_dim = 7, activation = "relu"))
model.add(Dense(units = 32, activation = "relu"))
model.add(Dense(units = 64, activation = "relu"))
model.add(Dense(units = 128, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(units = 2, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()

model.fit(x_train, y_train, batch_size = 128, epochs = 10)

score = model.evaluate(x_test, y_test)
print(score)
scoring=model.predict(x_test)
