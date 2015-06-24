from pyspark.context import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

def parsePoint(line):
    values = [float(x.strip()) for x in line.split(',')]
    return LabeledPoint(values[-1],values[1:10])

data = sc.textFile("heart_disease.csv")
data = sc.textFile("heart_disease.csv")
data = data.map(parsePoint)


(trainingData, testData) = data.randomSplit([0.7, 0.3])
model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={},
                                                 numIterations=30, maxDepth=4)


# This works too!
train = sc.textFile("train.csv")
def parsePoint(line):
    values = [float(x.strip()) for x in line.split(',')]
    return LabeledPoint(values[-1],values[:65])
train = train.map(parsePoint)
model = GradientBoostedTrees.trainClassifier(train, categoricalFeaturesInfo={},
                                                 numIterations=300, maxDepth=2,learningRate=0.1)
test = sc.textFile("test.csv")
test = test.map(parsePoint)
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test.count())
print(testErr)

import pandas as pd

# These examples actually seem to work
# https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/gradient_boosted_trees.py

# http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.info.txt
# row.names,sbp,tobacco,ldl,adiposity,famhist,typea,obesity,alcohol,age,chd
# chd (coronary heart disease) = target
# :%s/Absent/0/g


