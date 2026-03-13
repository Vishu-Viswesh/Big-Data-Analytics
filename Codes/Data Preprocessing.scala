import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, RegexTokenizer, StopWordsRemover}

val rawReviews = spark.read.json("hdfs://localhost:9000/amazon/reviews")
val cleanedData = rawReviews
  .filter(col("rating").isNotNull) 
  .withColumn("rating", col("rating").cast("double"))
  .withColumn("review_date", from_unixtime(col("timestamp")))
  .withColumn("month", month(col("review_date")))
val userCounts = cleanedData.groupBy("user_id").count().filter(col("count") >= 3)

// Perform an inner join across the cluster to drop "cold" users from the training set
val denseReviews = cleanedData.join(userCounts, Seq("user_id"), "inner")

val userIndexer = new StringIndexer()
  .setInputCol("user_id")
  .setOutputCol("userIndex")
  .setHandleInvalid("skip")
  .fit(denseReviews)

val itemIndexer = new StringIndexer()
  .setInputCol("asin")
  .setOutputCol("itemIndex")
  .setHandleInvalid("skip")
  .fit(denseReviews)

// Transform the DataFrame to include the new integer-based index columns
val indexedData = itemIndexer.transform(userIndexer.transform(denseReviews))
val productText = denseReviews.select("asin", "title")
  .filter(col("title").isNotNull)
  .dropDuplicates("asin")

// Lexical Tokenization: Split text strings into arrays of discrete words
val tokenizer = new RegexTokenizer()
  .setInputCol("title")
  .setOutputCol("words")
  .setPattern("\\W+") // Split strictly on non-alphanumeric characters

val tokenizedData = tokenizer.transform(productText)
val stopWordsRemover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")

val finalNLPData = stopWordsRemover.transform(tokenizedData)
  .filter(size(col("filtered")) > 0) // Drop records that became empty after stop-word removal