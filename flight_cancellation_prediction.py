from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

categoricalColumns = ["UNIQUE_CARRIER", "ORIGIN", "DEST"]
numericalColumns = ["DISTANCE"]

# Convert string categorical columns to indexed integers
indexers = [
	StringIndexer(inputCol=c, outputCol ="{0}_indexed".format(c))
	for c in categoricalColumns
]

# OneHot Encoding
encoders = [
	OneHotEncoder(
		inputCol=indexer.getOutputCol(),
		outputCol ="{0}_encoded".format(indexer.getOutputCol())
		)
	for indexer in indexers
]

# Assembler for categorical columns
assemblerCategorical = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol= "cat")
stages = indexers+encoders+ [assemblerCategorical]
pipelineCategorical = Pipeline(stages=stages)
df = pipelineCategorical.fit(df).transform(df)

# Assembler for Numerical columns
assemblerNumerical = VectorAssembler(inputCols = numericalColumns, outputCol = "num")
pipelineNumerical = Pipeline(stages = [assemblerNumerical])
df = pipelineNumerical.fit(df).transform(df)

# Assembler to combine both pipelines
assembler = VectorAssembler(inputCols = ["cat", "num"], outputCol = "features")
pipeline = Pipeline(stages=[assembler])
df = pipeline.fit(df).transform(df)

# Split data into train (70%) and test (30%)
train, test = df.randomSplit([0.7, 0.3], seed = 2018)

# Logistic Regression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'CANCELLED', maxIter= 10)
lrModel = lr.fit(train)
predictions_lr = lrModel.transform(test)

# Evaluating accuracy
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator().setLabelCol("CANCELLED")
accuracy_lr = evaluator.evaluate(predictions_lr)
# 0.68863895345779458

# Decision Tree
from pyspark.ml.evaluation import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="CANCELLED", featuresCol="features", maxDepth=3)
dtModel = dt.fit(train)
predictions_dt = dtModel.transform(test)
accuracy_dt = evaluator.evaluate(predictions_dt)
# 0.53058774985083534

# Random Forest
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="CANCELLED", featuresCol="features")
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)
accuracy_rf = evaluator.evaluate(predictions_rf)

# Naive Bayes
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial", featuresCol = "features", labelCol = "CANCELLED")
nbModel = nb.fit(train)
predictions_nb = nbModel.transform(test)
accuracy_nb = evaluator.evaluate(predictions_nb)
# 0.59431219823991344

# SVM - tried but didn't work
from pyspark.ml.classification import LinearSVC
lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol = ‘features’, labelCol = ‘CANCELLED’)
lsvcModel = lsvc.fit(train)
predictions_svm = lsvcModel.transform(test)
accuracy_svm = evaluator.evaluate(predictions_svm)

# Plotting accuracies for all models
import matplotlib.pyplot as plt
A = ['Logistic Regression','Decision Tree','Random Forest','Naive Bayes']
B = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_nb]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(A, B)
axes = plt.gca()
axes.set_ylim([0,100])

plt.show()
