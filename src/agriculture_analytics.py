"""
Smart Agriculture Data Analytics using Apache Spark.

Input: district-level Indian Area, Production and Yield (APY) CSV.
Run:
  spark-submit src/agriculture_analytics.py --input data/APY.csv --output outputs

The program performs distributed-style DataFrame cleaning, aggregation and
exploratory analytics with Spark, then compares Random Forest and Gradient-
Boosted Tree regressors for production estimation. Run mode depends on the
Spark environment; the supplied Colab notebook uses local Spark mode.
"""
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, sum as spark_sum, avg, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to APY CSV")
    parser.add_argument("--output", default="outputs", help="Output folder")
    args = parser.parse_args()

    spark = (SparkSession.builder
             .appName("SmartAgricultureBigDataAnalytics")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.read.option("header", True)
           .option("inferSchema", True)
           .csv(args.input))

    required = ["Year", "State Name", "District Name", "Crop Name",
                "Crop Code", "Crop Type", "Season", "Area", "Production", "Yield"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError("CSV is missing expected APY columns: " + ", ".join(missing))

    df = raw.select(
        col("Year").cast("string").alias("year"),
        col("State Name").cast("string").alias("state"),
        col("District Name").cast("string").alias("district"),
        col("Crop Name").cast("string").alias("crop"),
        col("Crop Code").cast("string").alias("crop_code"),
        col("Crop Type").cast("string").alias("crop_type"),
        col("Season").cast("string").alias("season"),
        col("Area").cast("double").alias("area"),
        col("Production").cast("double").alias("production"),
        col("Yield").cast("double").alias("yield")
    )
    df = (df.withColumn("year_start",
                        regexp_extract(col("year"), r"^(\\d{4})", 1).cast("int"))
          .dropna(subset=["year_start", "state", "district", "crop",
                          "crop_code", "crop_type", "season",
                          "area", "production"])
          .filter((col("area") > 0) & (col("production") >= 0)))

    total_rows = df.count()
    print(f"Clean records: {total_rows:,}")
    print(f"Distinct states/UTs: {df.select('state').distinct().count():,}")
    print(f"Distinct districts: {df.select('district').distinct().count():,}")
    print(f"Distinct crops: {df.select('crop_code').distinct().count():,}")
    print(f"Year range: {df.agg({'year_start':'min'}).first()[0]} - "
          f"{df.agg({'year_start':'max'}).first()[0]}")

    # Core Big Data Analytics: aggregate large district-level records.
    (df.groupBy("crop").agg(spark_sum("production").alias("total_production"),
                            spark_sum("area").alias("total_area"),
                            avg("yield").alias("average_yield"),
                            count("*").alias("records"))
       .orderBy(col("total_production").desc())
       .limit(20).write.mode("overwrite").option("header", True)
       .csv(args.output + "/top_crops"))

    (df.groupBy("state").agg(spark_sum("production").alias("total_production"),
                             spark_sum("area").alias("total_area"),
                             count("*").alias("records"))
       .orderBy(col("total_production").desc())
       .write.mode("overwrite").option("header", True)
       .csv(args.output + "/state_wise_production"))

    (df.groupBy("season").agg(spark_sum("production").alias("total_production"),
                              avg("yield").alias("average_yield"),
                              count("*").alias("records"))
       .orderBy(col("total_production").desc())
       .write.mode("overwrite").option("header", True)
       .csv(args.output + "/season_wise_analysis"))

    (df.groupBy("year_start").agg(spark_sum("production").alias("total_production"),
                                  spark_sum("area").alias("total_area"))
       .orderBy("year_start")
       .write.mode("overwrite").option("header", True)
       .csv(args.output + "/yearly_trends"))

    # Regression: estimate production from area and crop/geographic/seasonal context.
    categorical = ["state", "district", "crop", "crop_code", "crop_type", "season"]
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx",
                              handleInvalid="keep") for c in categorical]
    encoder = OneHotEncoder(
        inputCols=[c + "_idx" for c in categorical],
        outputCols=[c + "_vec" for c in categorical],
        handleInvalid="keep"
    )
    assembler = VectorAssembler(
        inputCols=["area", "year_start"] + [c + "_vec" for c in categorical],
        outputCol="features"
    )
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            labelCol="production", featuresCol="features",
            numTrees=30, maxDepth=8, seed=42
        ),
        "Gradient-Boosted Tree Regressor": GBTRegressor(
            labelCol="production", featuresCol="features",
            maxIter=20, maxDepth=5, seed=42
        )
    }
    rmse_eval = RegressionEvaluator(labelCol="production",
                                    predictionCol="prediction", metricName="rmse")
    r2_eval = RegressionEvaluator(labelCol="production",
                                  predictionCol="prediction", metricName="r2")
    metrics = []
    for name, model in models.items():
        pipeline = Pipeline(stages=indexers + [encoder, assembler, model])
        fitted = pipeline.fit(train)
        predictions = fitted.transform(test)
        rmse = float(rmse_eval.evaluate(predictions))
        r2 = float(r2_eval.evaluate(predictions))
        metrics.append((name, rmse, r2))
        print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")
        if name == "Random Forest Regressor":
            predictions.select("year", "state", "district", "crop",
                               "production", "prediction").limit(1000) \
                .write.mode("overwrite").option("header", True) \
                .csv(args.output + "/sample_predictions")

    metrics_df = spark.createDataFrame(metrics, ["model", "rmse", "r2"])
    metrics_df.orderBy(col("rmse").asc()).write.mode("overwrite") \
        .option("header", True).csv(args.output + "/model_metrics")
    print("\nAnalytics outputs written under:", args.output)
    print("Model comparison saved under:", args.output + "/model_metrics")
    spark.stop()


if __name__ == "__main__":
    main()
