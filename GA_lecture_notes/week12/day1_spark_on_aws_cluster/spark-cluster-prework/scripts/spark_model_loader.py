import pyspark as ps    
import warnings     
import sys
from pyspark.sql import SQLContext

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

try:
    # use local on your computer, yarn on the cluster
    sc = ps.SparkContext('local[4]')
    #sc = ps.SparkContext('yarn')
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    # give a warning if SparkContext already exists (for use inside pyspark)
    warnings.warn("SparkContext already exists in this scope")
    
evaluator = RegressionEvaluator(predictionCol='prediction',
                                labelCol='MEDV',
                                metricName='r2')

spark = ps.sql.SparkSession(sc)

spark_df = spark.read.csv(
    path='../data/boston_housing.csv', # for local use
    #path='s3://galdn-dsi3-babsdata/boston/boston_housing.csv', # for use on the cluster
    header=True,
    mode="DROPMALFORMED",
    inferSchema=True,
    enforceSchema=False
    )


best_model = PipelineModel.load(sys.argv[1])

predictions = best_model.transform(spark_df)

print(evaluator.evaluate(predictions))