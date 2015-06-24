from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import pylab as pl 
pl.gray() 
pl.matshow(digits.images[0]) 
pl.show()

# Talks about the Digits dataset
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
 
import pandas as pd


import pandas as pd
digits['data'][0]

# This stuff below should all work
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.ensemble import GradientBoostingClassifier as GBM
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
digits = load_digits()
predictors = pd.DataFrame(digits['data'])
target = pd.DataFrame(digits['target'])
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.33, random_state=42)
tune_parameters = [{'n_estimators':[10,100,150,200,300],
                    'learning_rate':[0.1,0.01],
                    'max_depth':[1,2,3]}]
gbm = grid_search.GridSearchCV(GBM(), tune_parameters,n_jobs=8).fit(X_train,np.ravel(y_train))
predictions = gbm.predict(X_test)
accuracy_score(y_test, predictions)

gbm.best_params_

df = pd.DataFrame(digits['data'])
df['target'] = digits['target']
train, test, useless1, useless2 = train_test_split(df, df, test_size=0.33, random_state=42)
train.to_csv(path_or_buf='train.csv',header=False)
test.to_csv(path_or_buf='test.csv',header=False)

X_train.to_csv(path_or_buf='X_train.csv')
X_test.to_csv(path_or_buf='X_test.csv')
y_train.to_csv(path_or_buf='y_train.csv')
y_test.to_csv(path_or_buf='y_test.csv')

# The Spark MLlib docs
# https://spark.apache.org/docs/latest/mllib-linear-methods.html



###

X_train = sc.textFile("X_train.csv")
X_test = sc.textFile("X_test.csv")
y_train = sc.textFile("y_train.csv")
y_test = sc.textFile("y_test.csv")



from pyspark.context import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.util import MLUtils
train = sc.textFile("train.csv")
test = sc.textFile("test.csv")
model = GradientBoostedTrees.trainClassifier(X_train, categoricalFeaturesInfo={},
                                                 numIterations=30, maxDepth=4)
###


#### Spark's MLlib
from __future__ import print_function

import sys

from pyspark.context import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.util import MLUtils


def testClassification(trainingData, testData):
    # Train a GradientBoostedTrees model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={},
                                                 numIterations=30, maxDepth=4)
    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1]).count() \
        / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification ensemble model:')
    print(model.toDebugString())


def testRegression(trainingData, testData):
    # Train a GradientBoostedTrees model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    model = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                                numIterations=30, maxDepth=4)
    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda vp: (vp[0] - vp[1]) * (vp[0] - vp[1])).sum() \
        / float(testData.count())
    print('Test Mean Squared Error = ' + str(testMSE))
    print('Learned regression ensemble model:')
    print(model.toDebugString())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Usage: gradient_boosted_trees", file=sys.stderr)
        exit(1)
    sc = SparkContext(appName="PythonGradientBoostedTrees")

    # Load and parse the data file into an RDD of LabeledPoint.
    data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    print('\nRunning example of classification using GradientBoostedTrees\n')
    testClassification(trainingData, testData)

    print('\nRunning example of regression using GradientBoostedTrees\n')
    testRegression(trainingData, testData)

    sc.stop()





from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
train = sc.textFile("train.csv")
# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[-1], values[:-1])

#data = sc.textFile("data/mllib/sample_svm_data.txt")
parsedData = train.map(parsePoint)

# Build the model
model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "myModelPath")
sameModel = SVMModel.load(sc, "myModelPath")
