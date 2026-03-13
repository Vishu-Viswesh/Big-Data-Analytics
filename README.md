# Big-Data-Analytics

# 🚀 Shadow OS: Distributed Big Data Recommendation Pipeline

**Shadow OS** is a large-scale, distributed hybrid recommender system designed to solve computational bottlenecks associated with highly sparse e-commerce datasets. By replacing memory-bound single-node frameworks with an **Apache Spark** and **Hadoop (HDFS)** backend, the system successfully processes **8.7 GiB** of interaction logs. 

---

## 🏗️ System Architecture

1. **Distributed Storage (HDFS):** Raw JSON logs (Amazon Electronics Dataset) are ingested, partitioned, and stored across Hadoop DataNodes.
2. **Parallel Computation (Spark MLlib):** A 3-executor, 12-core standalone Spark cluster reads the HDFS blocks in parallel, avoiding Out-Of-Memory (OOM) fragmentation.
3. **Hybrid Algorithm:** * **Path A (Behavioral):** Distributed Alternating Least Squares (ALS) matrix factorization.
   * **Path B (Semantic):** Word2Vec natural language embeddings to solve the 99.997% sparsity cold-start problem.
4. **Prescriptive Visualization (Streamlit & LLM):** The extreme data boundaries are coalesced, visualized in a Python dashboard, and passed into the Groq Cloud API for Llama 3.1 to generate actionable business directives.

---

## 📊 Big Data Performance Metrics

* **Input Data Volume:** 8.7 GiB (400K+ Rows)
* **Execution Environment:** 3-Executor Standalone Cluster (6 Active Cores, 4GB RAM per Executor)
* **Distributed Load Balancing:** 30 explicitly defined Shuffle Partitions
* **Execution Throughput:** 2,449 distributed DAG stages completed in 6.9 cumulative compute hours.

---

## 🛠️ Prerequisites

To run this pipeline locally or on a cluster, you need:
* **Hadoop 3.x** (Running NameNode and DataNodes)
* **Apache Spark 3.x** (Configured for Standalone Mode)
* **Scala 2.12+**
* **Python 3.8+**
* A **Groq Cloud API Key** (for Llama 3.1 inference)

---

## ⚙️ Installation & Execution

### 1. Start the Hadoop Cluster & Ingest Data
Ensure HDFS is running and upload your raw JSON dataset to the distributed blocks.
// start-dfs.sh
// hdfs dfs -mkdir -p /amazon/reviews
// hdfs dfs -put /local/path/to/amazon_electronics.json /amazon/reviews/

spark-submit \
  --class com.shadowos.Pipeline \
  --master spark://localhost:7077 \
  --executor-memory 4G \
  --total-executor-cores 12 \
  target/scala-2.12/shadow-os-assembly-1.0.jar

pip install -r requirements.txt
export GROQ_API_KEY="your_api_key_here"
streamlit run app.py
