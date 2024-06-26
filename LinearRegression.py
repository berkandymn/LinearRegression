#%%
#kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import statsmodels.api  as sm

#%%
#verilerin okunması
veriler=pd.read_csv('odev_tenis.csv')
veriler=veriler.sort_values(by=['temperature'],ascending=False)
print(veriler)

#%%
#fonksiyonlar
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
regressior=LinearRegression()
sc=StandardScaler()
lr=LinearRegression()

#%%
#verilerin ayrılması
windy=veriler.iloc[:,3:4].values
play=veriler.iloc[:,4:5].values

windy[:,0]=le.fit_transform(veriler.iloc[:,3])
windy=ohe.fit_transform(windy).toarray()

play[:,0]=le.fit_transform(veriler.iloc[:,4])
play=ohe.fit_transform(play).toarray()

temp=veriler.iloc[:,1:2].values
humid=veriler.iloc[:,2:3].values.astype(float)

outSide=veriler.iloc[:,0:1].values
outSide[:,0]=le.fit_transform(veriler.iloc[:,0:1])
outSide=ohe.fit_transform(outSide).toarray()


#%%
#ön hazırlığı yapılan verilerin birleştirilmesi
sonuc=pd.DataFrame(data=outSide,index=range(14),columns=['overcast','rainy','sunny'])
sonuc2=pd.DataFrame(data=temp,index=range(14),columns=['Temperature'])
sonuc3=pd.DataFrame(data=humid,index=range(14),columns=['humidity'])
sonuc4=pd.DataFrame(data=windy[:,1],index=range(14),columns=['windy'])
sonuc5=pd.DataFrame(data=play[:,1],index=range(14),columns=['Play'])

s1=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s1,sonuc3],axis=1)
s3=pd.concat([s2,sonuc4],axis=1)
s4=pd.concat([s3,sonuc5],axis=1)

#%%

sol=s4.iloc[:,0:3]
sag=s4.iloc[:,4:]

egitim=pd.concat([sol,sag],axis=1)

print(egitim)

#%%
x_train, x_test, y_train, y_test=train_test_split(egitim,sonuc2,test_size=0.33,random_state=0)

lr.fit(x_train,y_train)
yPredict=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
print(y_test)
print(yPredict)




# %%

X=np.append(arr=np.ones((14,1)).astype(int),values=egitim.iloc[:,0:],axis=1)

xListe=egitim.iloc[:,[0,1,2,3,4]].values
xListe=np.array(xListe,dtype=float)
model=sm.OLS(sonuc2,xListe).fit()
print(model.summary())

#%%

x_train=x_train.iloc[:,0:3]
x_test=x_test.iloc[:,0:3]

lr.fit(x_train,y_train)
yPredict=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
print(y_test)
print(yPredict)




# %%
