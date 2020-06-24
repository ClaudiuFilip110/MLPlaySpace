#import'uri
import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
#download data set from kaggle

#read from csv
dataset_filepath = "worldwide.csv"

covid = pd.read_csv(dataset_filepath)

#use only the data we want
data = covid[['Country/Region', 'Confirmed', 'Active', 'Recovered', '1 week change', '1 week % increase', 'Deaths','WHO Region']]

le = preprocessing.LabelEncoder()

Country = le.fit_transform(list(data['Country/Region']))
Confirmed = le.fit_transform(list(data['Confirmed']))
Active = le.fit_transform(list(data['Active']))
Recovered = le.fit_transform(list(data['Recovered']))
OneWeekChange = le.fit_transform(list(data['1 week change']))
OneWeekIncrease = le.fit_transform(list(data['1 week % increase']))
Deaths = le.fit_transform(list(data['Deaths']))
WhoRegion = le.fit_transform(list(data['WHO Region']))

recovered = le.fit(list(data['Recovered']))
whoRegion = le.fit(list(data['WHO Region']))

X = list(zip(Country, Confirmed, Active, Deaths, OneWeekIncrease))
y = list(zip(Recovered, WhoRegion))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=15)

model.fit(x_test, y_test)
acc = model.score(x_test, y_test)

predict = model.predict(x_test)

#change the type to int from float
countryPred = predict[:,1].astype(int)
#get the countries instead of ints
countries = recovered.inverse_transform(countryPred)
print('This model predicts the number of recovered people in each region:')
for i in range(len(x_test)):
    try:
        #predict the number of recovered people and the region in which they are
        print("Predicted: ", predict[i][0],', ', countries[int(predict[i][1])], "---- Actual: ", y_test[i][0], ', ', countries[int(y_test[i][1])])
    except NameError:
        print('an error has occured')
        
print('The accuracy of the model is ', acc*100, "%")
input()
