#import'uri
import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
#download data set from kaggle

#read from csv
dataset_filepath = 'canada-cumulative-case-count-by-new-hybrid-regional-health-boundaries.csv'

covid = pd.read_csv(dataset_filepath)

#use only the data we want
data = covid[['engname', 'casecount', 'recovered', 'totalpop2019', 'deaths']]

le = preprocessing.LabelEncoder()
engname = le.fit_transform(list(data['engname']))
casecount = le.fit_transform(list(data['casecount']))
recovered = le.fit_transform(list(data['recovered']))
totalpop2019 = le.fit_transform(list(data['totalpop2019']))
deaths = le.fit_transform(list(data['deaths']))

X = list(zip(engname, casecount, recovered, totalpop2019))
y = list(deaths)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=1)

model.fit(x_test, y_test)
acc = model.score(x_test, y_test)
print(acc)

predict = model.predict(x_test)

"""
for i in range(len(x_test)):
    print("Predicted: ", predict[i], " Data: ", x_test[i], " Actual: ", y_test[i])
    n = model.kneighbors([x_test[i]], 9)"""
