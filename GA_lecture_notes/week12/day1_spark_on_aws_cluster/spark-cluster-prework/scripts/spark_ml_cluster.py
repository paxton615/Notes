import pyspark as ps   
import warnings         
from pyspark.sql import SQLContext

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

try:
    # we try to create a SparkContext to work locally on all cpus available
    sc = ps.SparkContext('local[4]')
    # in the cluster script, replace with 
    #sc = ps.SparkContext('yarn')
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    # give a warning if SparkContext already exists (for use inside pyspark)
    warnings.warn("SparkContext already exists in this scope")
    
spark = ps.sql.SparkSession(sc)

# on the cluster, replace with the following
spark_df = spark.read.csv(
    #path='../data/boston_housing.csv', # for local use
    path='s3://galdn-dsi3-babsdata/boston/boston_housing.csv', # for use on the cluster
    header=True,
    mode="DROPMALFORMED",
    inferSchema=True,
    enforceSchema=False
    )

(data_train, data_test) = spark_df.randomSplit([0.7, 0.3], seed=1)

features = [col for col in spark_df.columns if col != 'MEDV']

vectorAssembler = VectorAssembler(inputCols=features,
                                  outputCol="features")

scaler = StandardScaler(withMean=True,
                        inputCol="features",
                        outputCol="scaledfeatures")

model = LinearRegression(featuresCol=scaler.getOutputCol(),
                         labelCol='MEDV',
                         maxIter=3000,
                         regParam=0.0,
                         elasticNetParam=0.0)

pipeline = Pipeline(stages=[vectorAssembler, scaler, model])

evaluator = RegressionEvaluator(predictionCol='prediction',
                                labelCol='MEDV',
                                metricName='r2')

reg_strengths = sc.range(-4, 4).map(lambda x: 10**x).collect()

paramGrid = ParamGridBuilder() \
    .addGrid(model.regParam, reg_strengths) \
    .addGrid(model.fitIntercept, [True, False]) \
    .build()

# the actual gridsearch
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5,
                          parallelism=2)

# Run cross-validation, and choose the best set of parameters.
model_fit = crossval.fit(data_train)


predictions_train = model_fit.transform(data_train)
predictions_test = model_fit.transform(data_test)
predictions_all = model_fit.transform(spark_df)

print(evaluator.evaluate(predictions_train))
print(evaluator.evaluate(predictions_test))
print(evaluator.evaluate(predictions_all))

java_model = model_fit.bestModel.stages[2]._java_obj
best_parameters = {param.name: java_model.getOrDefault(java_model.getParam(param.name))
       for param in paramGrid[0]}

print(best_parameters)
print()
print(java_model.explainParams())

import time
time_now = time.strftime("%Y%m%d-%H%M%S")
path = 'model-{}'.format(time_now)
model_fit.bestModel.save(path)

def add_line(text, file, n_lines=1):
    return file.write(str(text)+'\n'*n_lines)

with open('results-{}.txt'.format(time_now), 'w') as file:
    add_line(best_parameters, file, n_lines=2)
    add_line(java_model.explainParams(), file, n_lines=2)
    add_line(evaluator.evaluate(predictions_train), file)
    add_line(evaluator.evaluate(predictions_test), file)
    add_line(evaluator.evaluate(predictions_all), file)

