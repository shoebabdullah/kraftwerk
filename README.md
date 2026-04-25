# Capstone Project: Automated Extraction of Pros and Cons from Business Documents

# Module 20 Submission Required Capstone Assignment 20.1: Initial Report and Exploratory Data Analysis (EDA)

## Overview
Reading through long business documents to find the "good" and "bad" points takes too much time and often leads to people missing important details. This project builds a smart tool that automatically scans these documents to pull out a balanced list of Pros and Cons, helping people make faster, unbiased decisions.

* **Pros (The "Upside"):** Reasons to say "Yes" (benefits, gains, opportunities).
* **Cons (The "Downside"):** Reasons to say "No" (risks, costs, potential problems).

## Research Question
To what extent can a machine learning model be trained using unsupervised clustering and supervised ensemble methods to accurately extract and categorize "Pros" and "Cons" from unstructured business documents to improve decision-making efficiency?

## Dataset
The dataset consists of **3,000 unstructured text snippets** from business documents and reviews, labeled as either Pros (1) or Cons (0).
* **Cleaned Data:** 2,996 records (4 rows with corrupted/missing labels were removed).
* **Class Distribution:** The dataset is perfectly balanced with a ~50/50 split between Pros and Cons, mitigating class imbalance bias.

## Exploratory Data Analysis (EDA) & Feature Engineering
Before feeding the text into machine learning models, the unstructured text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorization.
* Common English stop words were removed.
* The feature space was limited to the top 1,500 most significant text elements to reduce noise.

## Modeling & Results

### 1. Supervised Baseline Model (Logistic Regression)
A Logistic Regression model was trained as a baseline to establish a benchmark. 
* **Training Split:** 80% Train / 20% Test
* **Overall Accuracy:** **83.00%**
* **Cons (0):** Precision: 83% | Recall: 85%
* **Pros (1):** Precision: 83% | Recall: 81%

*Insight:* An 83% accuracy score using a simple linear model indicates a strong signal in the dataset, proving that machine learning can accurately extract and categorize these text segments.

### 2. Unsupervised Pattern Discovery (PCA & K-Means)
To observe natural groupings blindly (without using labels), an unsupervised pipeline was applied:
* **PCA (Principal Component Analysis):** Reduced the 1500-dimensional TF-IDF space down to 2 principal components for visualization.
* **K-Means Clustering:** Segmented the data into exactly 2 natural clusters based on mathematical similarities.

*Insight:* While K-Means cleanly splits the space into two clusters, comparing this to the true labels reveals a dense region of overlap in the center. Because both positive and negative points share common business terminology (e.g., "product interface"), simple word counts blend together. 

*(Note: Add the generated `class_distribution.png`, `pca_kmeans_clusters.png`, and `pca_true_labels.png` from your EDA to your repository and embed them in this section).*

## Next Steps & Future Work
To improve the tool and fully address the research question, the next phases of the project will focus on:
1. **Contextual Embeddings:** Transitioning from TF-IDF to contextual deep learning embeddings (e.g., BERT, Word2Vec) to better understand word associations and disambiguate overlapping terminology.
2. **Ensemble & Advanced Methods:** Utilizing algorithms like Random Forests, XGBoost, or Deep Neural Networks to capture non-linear relationships that the baseline model may have missed.


## Expanded Model Comparison
To further investigate the research question, we implemented several advanced supervised learning algorithms. Below is a comparison of their performance:

| Model                |   Accuracy |   Precision |   Recall |   F1-Score |
|:---------------------|-----------:|------------:|---------:|-----------:|
| Logistic Regression  |   0.83     |    0.825455 | 0.807829 |   0.816547 |
| Random Forest        |   0.823333 |    0.848606 | 0.758007 |   0.800752 |
| Lasso (L1 Logistic)  |   0.798333 |    0.838983 | 0.704626 |   0.765957 |
| Gradient Boosting    |   0.798333 |    0.88835  | 0.651246 |   0.75154  |
| Decision Tree        |   0.765    |    0.763158 | 0.72242  |   0.74223  |
| Neural Network (MLP) |   0.74     |    0.716263 | 0.736655 |   0.726316 |

*Results show that Logistic Regression performed the best with an accuracy of 83.00%.*
