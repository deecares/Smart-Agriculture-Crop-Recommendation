# Smart Agriculture: Crop Recommendation Using Apache Spark

A mini Big Data Analytics project that uses **Apache Spark MLlib (PySpark)** to classify the most suitable crop from soil and weather measurements.

## 1. Problem statement

Agricultural crop suitability depends on soil nutrients and environmental conditions. This project trains multiclass classification models on those measurements and compares their performance on a held-out test set.

**Selected mini case study:** 11.4 – Smart Agriculture Data Analytics Platform  
**Machine-learning problem:** Multiclass classification  
**Big Data technology:** Apache Spark / PySpark / Spark MLlib

## 2. Dataset

- **Dataset:** Crop Recommendation Dataset
- **Original platform / contributor:** Kaggle, Atharva Ingle
- **Dataset page:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Records:** 2,200
- **Input features:** 7
- **Target:** `label` (crop name; 22 categories)
- **Columns:** `N, P, K, temperature, humidity, ph, rainfall, label`

The Kaggle dataset describes N, P and K soil nutrient measurements, temperature, relative humidity, soil pH and rainfall. Use the original Kaggle page for dataset attribution and licensing details.

## 3. Models compared

The Spark MLlib pipeline trains and evaluates:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Logistic Regression (multinomial classification)

The script uses an 80:20 random train/test split with a fixed seed and reports test accuracy for each model. **The best model is selected from the actual run output. No accuracy or winning model is claimed in advance.**

## 4. Run in Google Colab (recommended for a Windows PC)

1. Open the notebook: [Run_on_Google_Colab.ipynb](notebooks/Run_on_Google_Colab.ipynb).
2. Click **Open in Colab** or download the notebook and upload it to https://colab.research.google.com/.
3. Run the cells from top to bottom.
4. The notebook installs Java and PySpark, downloads the dataset, runs the Spark script and prints model test accuracies.
5. Record the measured best model and accuracy in your lab submission.

Google Colab is a convenient cloud notebook environment; it is separate from the college's shared Hadoop server. This project demonstrates Spark MLlib locally within the Colab runtime, not a distributed multi-node Spark cluster.

## 5. Run on a machine that already has Spark

Install dependencies from `requirements.txt`, then run:

```bash
spark-submit src/crop_recommendation.py \
  --input data/Crop_recommendation.csv \
  --output outputs/model_metrics
```

The CSV must have the eight columns listed above. The output directory contains a CSV folder with model names and measured test accuracies.

## 6. Repository structure

```text
Smart-Agriculture-Crop-Recommendation/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── Run_on_Google_Colab.ipynb
└── src/
    └── crop_recommendation.py
```

## 7. Submission details

| Submission field | Project detail |
|---|---|
| Selected ML problem type | Classification |
| BDA project details | Apache Spark (PySpark / Spark MLlib) |
| Dataset name | Crop Recommendation Dataset |
| Dataset source platform | Kaggle |
| Dataset URL | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| ML models used | Decision Tree, Random Forest, Logistic Regression |
| Best-performing model | Fill in after running the notebook and comparing test accuracy |
| GitHub source code | https://github.com/deecares/Smart-Agriculture-Crop-Recommendation |
| LinkedIn post | Add after publishing a project post |

## 8. Scope and limitations

This is an educational crop classification demonstration based on the public dataset. Model predictions should not be treated as professional agronomic advice. Real deployment would need locally validated soil tests, regional and seasonal information, current weather, and guidance from agricultural experts.
