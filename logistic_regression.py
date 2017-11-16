'''
Spark job for regression analysis
'''

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import MinMaxScaler, RFormula
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .appName("PythonPi") \
    .getOrCreate()

# Import Data
dta = spark.read.csv("data/HOF_tr.csv", header=True, inferSchema=True)

# Assemble the features using R notation
formula = RFormula(formula="HOF ~ .", featuresCol="features", labelCol="label")
dta_features = formula.fit(dta).transform(dta).select("label", "features")

# Scale the data
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
input_data = scaler.fit(dta_features).transform(dta_features).select('label', 'scaledFeatures')

# Split data into testing and training sets
training, testing = input_data.randomSplit([.8, .2], seed = 1234)

# Logistic Regression Estimator
lr = LogisticRegression(
    maxIter=10,
    regParam=0.3,
    elasticNetParam=0.8,
    featuresCol="scaledFeatures"
)

# Parameter Grid to Search
pg = (ParamGridBuilder()
    .addGrid(lr.regParam, [0, .5, 1, 2])
    .addGrid(lr.elasticNetParam, [0, .5, 1])
    .build()
    )

# Prediction Evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")

# Cross Validation
cv = CrossValidator(
    estimator = lr,
    estimatorParamMaps = pg,
    evaluator = evaluator,
    numFolds = 4)

# Model fitting
cv_model = cv.fit(training)

show_columns = ['label', 'prediction', 'rawPrediction', 'probability']

# Prediction on Training Data
pred_training_cv = cv_model.transform(training)
pred_training_cv.select(show_columns).show(4, truncate = False)

# Prediction on Testing Data
pred_test_cv = cv_model.transform(testing)
pred_test_cv.select(show_columns).show(4, truncate = False)

pred_test_cv.groupBy('label', 'prediction').count().show()
