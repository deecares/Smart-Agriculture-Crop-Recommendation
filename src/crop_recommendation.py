"""
Smart Agriculture: Crop Recommendation using Apache Spark MLlib.

Input CSV columns:
N, P, K, temperature, humidity, ph, rainfall, label

Run:
  spark-submit src/crop_recommendation.py --input data/Crop_recommendation.csv --output outputs/model_metrics

This script trains Decision Tree, Random Forest, and Logistic Regression
classifiers and reports held-out test accuracy. It does not assume a winner:
the best model is selected from the measured test accuracies.
"""
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    LogisticRegression,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path accessible to Spark")
    parser.add_argument("--output", default="outputs/model_metrics", help="Output directory")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("SmartAgricultureCropRecommendation")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(args.input, header=True, inferSchema=True)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Expected features: " + ", ".join(FEATURES) + ", label"
        )

    # Remove incomplete rows and assemble seven numeric measurements.
    df = df.select(*(FEATURES + ["label"])).dropna()
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    label_indexer = StringIndexer(
        inputCol="label", outputCol="label_index", handleInvalid="error"
    )

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="accuracy"
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            labelCol="label_index", featuresCol="features", seed=42
        ),
        "Random Forest": RandomForestClassifier(
            labelCol="label_index", featuresCol="features",
            numTrees=80, seed=42
        ),
        "Logistic Regression": LogisticRegression(
            labelCol="label_index", featuresCol="features", maxIter=100
        ),
    }

    results = []
    for name, classifier in models.items():
        pipeline = Pipeline(stages=[assembler, label_indexer, classifier])
        fitted = pipeline.fit(train)
        predictions = fitted.transform(test)
        accuracy = evaluator.evaluate(predictions)
        results.append((name, accuracy))
        print(f"{name} test accuracy: {accuracy:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\nModel comparison (held-out test set):")
    for name, accuracy in results:
        print(f"{name}: {accuracy:.4f}")

    best_name, best_accuracy = results[0]
    print(f"\nBest model on this run: {best_name} ({best_accuracy:.4f})")

    # Save actual measured metrics as a small CSV output.
    metric_rows = [(name, float(acc)) for name, acc in results]
    metrics_df = spark.createDataFrame(metric_rows, ["model", "test_accuracy"])
    metrics_df.coalesce(1).write.mode("overwrite").option("header", True).csv(args.output)
    print(f"Metrics saved to: {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
