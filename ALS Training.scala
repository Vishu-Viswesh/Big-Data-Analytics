import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator

val reviews = spark.read.json("hdfs://localhost:9000/amazon/reviews").limit(450000)   
val cleanedReviews = reviews
  .filter(col("rating").isNotNull)
  .withColumn("rating", col("rating").cast("double"))

// Keeping your previous threshold
val userCounts = cleanedReviews.groupBy("user_id").count().filter(col("count") >= 3)
val denseReviews = cleanedReviews.join(userCounts, Seq("user_id"), "inner")

val userIndexer = new StringIndexer().setInputCol("user_id").setOutputCol("user_index").setHandleInvalid("keep")
val itemIndexer = new StringIndexer().setInputCol("asin").setOutputCol("item_index").setHandleInvalid("keep")

val userIndexed = userIndexer.fit(denseReviews).transform(denseReviews)
val finalDF = itemIndexer.fit(userIndexed).transform(userIndexed)

val Array(training, test) = finalDF.randomSplit(Array(0.8, 0.2), seed = 1234L)        

// MODIFIED HYPERPARAMETERS TO STOP OVERFITTING
val als = new ALS()
  .setMaxIter(10)
  .setRegParam(0.35) // Increased heavily to penalize memorization
  .setRank(5)        // Decreased heavily to force generalization
  .setNonnegative(true)
  .setUserCol("user_index")
  .setItemCol("item_index")
  .setRatingCol("rating")
  .setColdStartStrategy("drop")

val model = als.fit(training)

val trainPredictions = model.transform(training)
val testPredictions = model.transform(test)

val rmseEval = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val maeEval = new RegressionEvaluator().setMetricName("mae").setLabelCol("rating").setPredictionCol("prediction")

println("--- PRESENTATION METRICS ---")
println("Training RMSE = " + rmseEval.evaluate(trainPredictions))
println("Testing RMSE = " + rmseEval.evaluate(testPredictions))
println("Training MAE = " + maeEval.evaluate(trainPredictions))
println("Testing MAE = " + maeEval.evaluate(testPredictions))
