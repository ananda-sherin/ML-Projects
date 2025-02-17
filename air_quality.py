import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'/content/city_day.csv')
df = df.dropna(subset=['AQI_Bucket'])
df = df.drop(['Date','PM10','NOx','NH3','Toluene','Xylene'],axis=1)
df = df.fillna({'PM2.5':0,'NO':0, 'NO2':0,'CO':0,'SO2':0,'O3':0,'Benzene':0,'AQI':0})
df

X = df.drop(['AQI_Bucket'],axis=1)
Y = df['AQI_Bucket']
X
Y

from sklearn.preprocessing import LabelEncoder
city = LabelEncoder()

X['City'] = city.fit_transform(X['City'])
X['City']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
stx = StandardScaler()

x_train = stx.fit_transform(x_train)
x_test = stx.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10,criterion='entropy')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

y_pred

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)
print(cm)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

def predict_air_quality():
  city = input('Enter City:')
  PM25 = input('Enter PM2.5:')
  NO = input('Enter NO:')
  NO2 = input('Enter NO2:')
  CO = input('Enter CO:')
  SO2 = input('Enter SO2:')
  O3 = input('Enter O3:')
  Benzene = input('Enter Benzene:')
  AQI = input('Enter AQI:')
  try:
    city = int(city)
    PM25 = float(PM25)
    NO = float(NO)
    NO2 = float(NO2)
    CO = float(CO)
    SO2 = float(SO2)
    O3 = float(O3)
    Benzene = float(Benzene)
    AQI = float(AQI)
  except ValueError:
    print('Invalid Input')
  new_data = pd.DataFrame({'City':[city],'PM2.5':[PM25],'NO':[NO],'NO2':[NO2],'CO':[CO],'SO2':[SO2],'O3':[O3],'Benzene':[Benzene],'AQI':[AQI]})
  new_data
  prediction = classifier.predict(new_data)
  print(prediction)

predict_air_quality()
