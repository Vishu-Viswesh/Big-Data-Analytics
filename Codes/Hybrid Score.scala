package com.shadowos

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.feature.{StringIndexer, RegexTokenizer, StopWordsRemover, Word2Vec}

object Pipeline {
  def main(args: Array[String]): Unit = {
    
    // 1. Initialize Distributed Cluster Architecture
    val spark = SparkSession.builder()
      .appName("ShadowOS_Triple_Executor_Engine")
      .config("spark.executor.instances", "3") 
      .config("spark.executor.cores", "4")
      .config("spark.executor.memory", "4g")
      .config("spark.sql.shuffle.partitions", "30")
      .getOrCreate()

    import spark.implicits._

    // 2. HDFS Ingestion & Preprocessing
    val rawReviews = spark.read.json("hdfs://localhost:9000/amazon/reviews")
    
    val cleanedData = rawReviews.filter(col("rating").isNotNull && col("title").isNotNull)
      .withColumn("rating", col("rating").cast("double"))
      .withColumn("month", month(from_unixtime(col("timestamp"))))
      .repartition(30)

    // Density Filtering (Mitigating 99.997% sparsity)
    val userCounts = cleanedData.groupBy("user_id").count().filter(col("count") >= 3)
    val denseReviews = cleanedData.join(userCounts, Seq("user_id"), "inner")

    // 3. Categorical Indexing for ALS
    val userIndexer = new StringIndexer().setInputCol("user_id").setOutputCol("userIndex").setHandleInvalid("skip").fit(denseReviews)
    val itemIndexer = new StringIndexer().setInputCol("asin").setOutputCol("itemIndex").setHandleInvalid("skip").fit(denseReviews)
    val indexedData = itemIndexer.transform(userIndexer.transform(denseReviews))

    // 4. ALS Matrix Factorization (Behavioral)
    val als = new ALS().setMaxIter(5).setRegParam(0.1).setUserCol("userIndex").setItemCol("itemIndex").setRatingCol("rating").setColdStartStrategy("drop")
    val alsModel = als.fit(indexedData)
    val alsPredictions = alsModel.transform(indexedData).withColumnRenamed("prediction", "ALS_Predicted_Rating")

    // 5. Word2Vec Semantic Embedding (NLP)
    val productText = denseReviews.select("asin", "title").dropDuplicates("asin")
    val tokenizer = new RegexTokenizer().setInputCol("title").setOutputCol("words").setPattern("\\W+")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    
    val tokenized = stopWordsRemover.transform(tokenizer.transform(productText))
    val w2v = new Word2Vec().setInputCol("filtered").setOutputCol("product_vector").setVectorSize(10).setMinCount(1).setMaxIter(2)
    val w2vModel = w2v.fit(tokenized)
    
    // Note: For full pipeline, calculate Cosine Similarity from product_vector to get BERT_Content_Score
    // Mocking semantic score & sales volume for pipeline completion
    val hybridInput = alsPredictions.withColumn("BERT_Content_Score", rand() * 5)
                                    .withColumn("Sales_Volume", rand() * 1000)

    // 6. Hybrid Success Score Calculation
    val finalDF = hybridInput.withColumn(
      "Hybrid_Success_Score",
      (col("ALS_Predicted_Rating") * 0.5) + (col("BERT_Content_Score") * 0.3) + (log(col("Sales_Volume") + 1) * 0.2)
    )

    // 7. Extract Top 10 & Bottom 5 via Windowing
    val windowTop = Window.partitionBy("month").orderBy(desc("Hybrid_Success_Score"))
    val windowBottom = Window.partitionBy("month").orderBy(asc("Hybrid_Success_Score"))

    val top10 = finalDF.withColumn("rank", rank().over(windowTop)).filter(col("rank") <= 10).withColumn("Category", lit("TOP_LEADER"))
    val bottom5 = finalDF.withColumn("rank", rank().over(windowBottom)).filter(col("rank") <= 5).withColumn("Category", lit("BOTTOM_RISK"))

    val outputDF = top10.union(bottom5).select("Category", "rank", "asin", "ALS_Predicted_Rating", "Hybrid_Success_Score")

    // 8. Coalesce and Write to HDFS
    outputDF.coalesce(1)
      .write.mode("overwrite").option("header", "true")
      .csv("hdfs://localhost:9000/results/shadow_os_final_report")

    spark.stop()
  }
}