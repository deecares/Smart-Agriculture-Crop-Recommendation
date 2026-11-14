"""
Smart Agriculture Data Analytics using Apache Spark.

Dataset: India district-wise crop production statistics (1997-2015).
Run:
  spark-submit src/agriculture_analytics.py --input data/crop_production.csv --output outputs

Uses Spark DataFrames for cleaning, aggregation and MLlib regression.
The Colab notebook runs Spark in local mode, not on a multi-node cluster.
"""
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, count, trim, lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to crop_production.csv")
    parser.add_argument("--output", default="outputs", help="Output folder")
    args = parser.parse_args()

    spark = (SparkSession.builder.appName("SmartAgricultureDataAnalytics")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.read.option("header", True).option("inferSchema", True)
           .option("mode", "DROPMALFORMED").csv(args.input))

    aliases = {
        "State_Name": "state", "District_Name": "district",
        "Crop_Year": "year", "Season": "season", "Crop": "crop",
        "Area": "area", "Production": "production"
    }
    missing = [c for c in aliases if c not in raw.columns]
    if missing:
        raise ValueError("CSV is missing expected columns: " + ", ".join(missing))

    df = raw.select(
        *[(col(src).cast("int").alias(dst)
           if dst == "year"
           else trim(col(src).cast("string")).alias(dst))
          for src, dst in aliases.items()
          if dst not in ("area", "production")],
        col("Area").cast("double").alias("area"),
        col("Production").cast("double").alias("production")
    ).dropna(subset=["state", "district", "year", "season", "crop",
                     "area", "production"])
    df = (df.filter((col("area") > 0) & (col("production") >= 0))
          .dropDuplicates())

    # Coconut values in the source are reported in inconsistent units;
    # exclude them from comparable production totals and model training.
    df = df.filter(lower(col("crop")) != "coconut")

    total_rows = df.count()
    print(f"Clean records after removing invalid rows and Coconut: {total_rows:,}")
    print(f"States/UTs: {df.select('state').distinct().count():,}")
    print(f"Districts: {df.select('district').distinct().count():,}")
    print(f"Crops: {df.select('crop').distinct().count():,}")
    print(f"Year range: {df.agg({'year':'min'}).first()[0]} - "
          f"{df.agg({'year':'max'}).first()[0]}")

    (df.groupBy("crop").agg(spark_sum("production").alias("total_production"),
                            spark_sum("area").alias("total_area"),
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
                              avg("production").alias("average_production"),
                              count("*").alias("records"))
       .orderBy(col("total_production").desc())
       .write.mode("overwrite").option("header", True)
       .csv(args.output + "/season_wise_analysis"))

    (df.groupBy("year").agg(spark_sum("production").alias("total_production"),
                            spark_sum("area").alias("total_area"),
                            count("*").alias("records"))
       .orderBy("year").write.mode("overwrite").option("header", True)
       .csv(args.output + "/yearly_trends"))

    # Estimate production from area and contextual features.
    categorical = ["state", "district", "season", "crop"]
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx",
                              handleInvalid="keep") for c in categorical]
    encoder = OneHotEncoder(
        inputCols=[c + "_idx" for c in categorical],
        outputCols=[c + "_vec" for c in categorical],
        handleInvalid="keep"
    )
    assembler = VectorAssembler(
        inputCols=["area", "year"] + [c + "_vec" for c in categorical],
        outputCol="features"
    )
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    # Use a compact Random Forest configuration for Colab's limited driver memory.
    # The previous GBT stage exhausted the available Spark/JVM resources.
    models = [
        ("Random Forest Regressor", RandomForestRegressor(
            labelCol="production", featuresCol="features",
            numTrees=10, maxDepth=6, seed=42))
    ]
    rmse_eval = RegressionEvaluator(labelCol="production",
                                    predictionCol="prediction", metricName="rmse")
    r2_eval = RegressionEvaluator(labelCol="production",
                                  predictionCol="prediction", metricName="r2")
    metrics = []
    for name, model in models:
        pipeline = Pipeline(stages=indexers + [encoder, assembler, model])
        fitted = pipeline.fit(train)
        predictions = fitted.transform(test)
        rmse = float(rmse_eval.evaluate(predictions))
        r2 = float(r2_eval.evaluate(predictions))
        metrics.append((name, rmse, r2))
        print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")
        if name == "Random Forest Regressor":
            (predictions.select("year", "state", "district", "season", "crop",
                                "area", "production", "prediction")
             .limit(1000).write.mode("overwrite").option("header", True)
             .csv(args.output + "/sample_predictions"))

    spark.createDataFrame(metrics, ["model", "rmse", "r2"]) \
        .orderBy(col("rmse").asc()).write.mode("overwrite") \
        .option("header", True).csv(args.output + "/model_metrics")
    print("\nAnalytics outputs written under:", args.output)
    spark.stop()


if __name__ == "__main__":
    main()
