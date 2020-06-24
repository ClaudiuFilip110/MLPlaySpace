# Machine Learning project
applying ML algorithms for Covid-19 datasets

The dataset used for this application is: UNCOVER COVID-19 Challenge from Kaggle.
 - https://www.kaggle.com/roche-data-science-coalition/uncover?
 - and another dataset from kaggle available in the repository
```
dataset_filepath = 'canada-cumulative-case-count-by-new-hybrid-regional-health-boundaries.csv'

covid = pd.read_csv(dataset_filepath)
```
The objective of this project is to predict the probability of you dying in a canadian hospital of Covid-19 and highlight the use of ML algorithms(K-Nearest neighbor, Linear Regression) to solve real world problems.

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

<h2> K Nearest Neighbors </h2>
I used a K-Nearest Neighbors model which looks at the K nearest points in the dataset and puts the tests in their respective places.
<h2> Accuracy and scoring the model </h2>
The accuray of my model was poor: only 0.6143396226415094. An accuracy of 61% is not great, but it is better than the next model: Linear Regression.

<h2>Changing the number of neighbors and test sizes</h2>
This being a rather scattered dataset(look at the lower graph) having a small number of neighbors helps with accuracy. By lowering the number of neighbors the accuracy of the model increased dramatically.

 - 9 neighbors results in accuracy 37.5%
 - 7 neighbors results in accuracy 41.4%
 - 3 neighbors results in accuracy 61.4%
 - 1 neighbor results in accuracy 96.0%
 
 Moreover, changing the test size from 20% to 10% increases the accuracy of the model even further to 98.3%




<br><br><br>
<h2>Linear Regression</h2>
Seeing how my KNN model didn't work very well with the provided dataset I tried to use Linear Regression to predict the number of deaths. This may not be such a good idea because the data is not in linear, but I will try it nonetheless. 

<h2> Using a plot </h2>
I used a plot to visualize the data.


![Graph](https://github.com/ClaudiuFilip110/MLPlaySpace/blob/master/graph.PNG)

```
import matplotlib.pyplot as pyplot
from matplotlib import style

style.use("ggplot")
pyplot.scatter(recovered, deaths)
pyplot.xlabel("recovered")
pyplot.ylabel("Deaths")
pyplot.show()
```

With this we can visualize the non-linearity of the graph and so we can understand the next point which is:

<h2>Accuracy</h2>
The accuracy of this model was 0.21 or 21% which means this model is unusable.

# Decision Tree Regression

Unlike the other two projects, this model has 2 labels and uses a different dataset in which we have

### features
 - `Country/Region, Confirmed, Active, 1 week change, 1 week % increase, Deaths`

### labels
 -  `Recovered, WHO Region( the biggest region nearby)`

This model has a few parameters the let you define the depth of the tree. For example is I chose the `max_leaf_nodes` to be 5 the accuracy of the model dropped to about 50% and as this number increased so did the accuracy. This is because of the way trees work. The depth of the tree depends on the number of the `max_leaf_nodes` so I chose the number as to not underfit or overfit.

In the end the `max_leaf_nodes` propriety was set to x and the accuracy of the model was 92.57%

