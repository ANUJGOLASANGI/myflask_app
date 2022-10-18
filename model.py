import pandas as pd
'''

KNN  the lazy algoithm

its  used for both classification and  regression problem
a lil inclined towards classification


'''

from sklearn.neighbors import KNeighborsClassifier
#KNeignbousRegressor for regression problem
from sklearn.model_selection import train_test_split
import pickle#for dumping th e model in the pickel var.

df=pd.read_csv('pizza.csv')
# print(df.head())

#to split into x and y   axis 
x=df.iloc[:,:-1]
y=df.iloc[:,-1]#index location

#for trainingthe  model
#ie splitting the data set into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

#calling the algorithm 
model=KNeighborsClassifier(n_neighbors=3)

#train the model
model.fit(x_train,y_train)

#for prediction/testing

# pred=model.predict([[25,69]])
# print(pred)


pickle.dump(model,open('model.pkl','wb'))

