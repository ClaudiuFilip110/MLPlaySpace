# Machine Learning small project
applying ML algorithms for Covid-19 datasets

The dataset used for this application is: UNCOVER COVID-19 Challenge from Kaggle.
 - https://www.kaggle.com/roche-data-science-coalition/uncover?
```
dataset_filepath = 'canada-cumulative-case-count-by-new-hybrid-regional-health-boundaries.csv'

covid = pd.read_csv(dataset_filepath)
```
The objective of this project is to highlight the use of ML algorithms(K-Means clustering) to solve real world problems.

The dataset provides a lot of columns, but I chose to work only with a small number of them:
- attributes: province, casecount, recovered, totalPop2019
- label: deaths

```
data = covid[['engname', 'casecount', 'recovered', 'totalpop2019', 'deaths']]
```

By choosing this label I want to see if the algorithm can correctly predict the number of deaths from a certain hospital:<br>
i. e. City of Hamilton Health Unit hospital has 19 deaths and the algorithm will predict x deaths. I will then score the algorithm based on how it's done.

<h2> Preprocessing </h2>
I've done some preprocessing before using the data so that we don't have to deal with the missing values.

```
from sklearn import preprocessing
e = preprocessing.LabelEncoder()
engname = le.fit_transform(list(data['engname']))
casecount = le.fit_transform(list(data['casecount']))
recovered = le.fit_transform(list(data['recovered']))
totalpop2019 = le.fit_transform(list(data['totalpop2019']))
deaths = le.fit_transform(list(data['deaths']))
```
I used the label encoder for the numbers as well so that the encoder can deal with the missing values on it's own.
<h2> Splitting the data </h2>
I've split the data into training and testing data so that I can see how accurate my model was. 

```
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
```
I've used a test size of 10% - 59% accuray, but those results were poorer than the ones with 20% test size who scored 61% accuracy.

<h2> Using the KNN model </h2>
I used a K-Nearest Neighbors model which looks at the K nearest points in the dataset and puts the tests in their respective places.
<h2> Accuracy and scoring the model </h2>
The accuray of my model was poor: only 0.6143396226415094 so I used a different model.
