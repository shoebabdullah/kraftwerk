# Required Capstone Project Module 24.1 - Final Report : Automated Extraction of Strategic Advantages (Pros) and Operational Risks (Cons) from Business Documents Using Machine Learning.
## Overview
### Reading through long business documents to find the "good" and "bad" points takes too much time and often leads to people missing important details. This project builds a smart tool that automatically scans these documents to pull out a balanced list of Pros and Cons, helping people make faster, unbiased decisions.

* **Pros (The "Upside"):** Reasons to say "Yes" (benefits, gains, opportunities).
* **Cons (The "Downside"):** Reasons to say "No" (risks, costs, potential problems).

### Research Question
To what extent can a machine learning model be trained using unsupervised clustering and supervised ensemble methods to accurately extract and categorize "Pros" and "Cons" from unstructured business documents to improve decision-making efficiency?

### Business Problem

Organizations generate large volumes of qualitative information through:
* Annual reports
* Earnings statements
* Investor presentations
* Corporate disclosures
* Strategic planning documents

### Manually reviewing these documents is time-consuming and subjective.

This project demonstrates how machine learning and NLP techniques can automatically:

* Classify positive and negative business statements
* Detect key themes within text
* Compare predictive model performance
* Extract strategic opportunities and risks from financial reports

## Executive Summary :  This capstone project develops an end-to-end Natural Language Processing (NLP) and Machine Learning pipeline that analyzes business and financial text to identify positive and negative sentiment, compare machine learning models, discover hidden text patterns through clustering, and generate strategic insights from external PDF documents.

### The system combines:

* Text preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
*  Supervised machine learning classification
*  Hyperparameter optimization
*  Unsupervised learning
*  PDF document parsing
*  Automated strategic pros and cons extraction
*  Data visualization and reporting

The primary objective is to transform unstructured business narratives into actionable insights that help identify organizational strengths, opportunities, risks, and challenges.

## Technologies Used
* **Programming Language :** Python
* **Data Processing :** NumPy, Pandas
* **Machine Learning :  Scikit-Learn**
* **Model Selection  :  train_test_split, GridSearchCV, cross_val_score**
* **Linear Model     :  LogisticRegression**
* **Tree             :  DecisionTreeClassifier**
* **Ensemble         :  RandomForestClassifier, GradientBoostingClassifier**
* **Decomposition    :  PCA**
* **Cluster          :  KMeans**
* **Metrics          :  Accuracy_score, Precision_score, Recall_score, F1_score**
* **Natural Language Processing :** TF-IDF Vectorization
* **Data Visualization :** Matplotlib, Seaborn, WordCloud
* **Document Processing :** PyPDF

## Link to Jupyter Notebook File : https://github.com/shoebabdullah/kraftwerk/blob/main/Capstone%20Project%2024.1%20Final%20Report%20v1.ipynb

## Project Architecture
<img width="348" height="722" alt="image" src="https://github.com/user-attachments/assets/c21e483c-bb4a-4553-95ab-ba07ddc70bc4" />

## Dataset : CapstoneDataSet.csv (Used for training the machine learning model) https://github.com/shoebabdullah/kraftwerk/blob/main/CapstoneDataSet.csv
The dataset consists of **3,000 unstructured text snippets** from business documents and reviews, labeled as either Pros (1) or Cons (0).
* **Cleaned Data:** 2,996 records (4 rows with corrupted/missing labels were removed).
* **Class Distribution:** The dataset is perfectly balanced with a ~50/50 split between Pros and Cons, mitigating class imbalance bias.

| Text               | Label |
| ------------------ | ----- |
| Positive statement | 1     |
| Negative statement | 0     |

Example : 
| Statement                                       | Value |
| ------------------------------------------------| ------| 
| Our company achieved record growth              | 1     |
| Inflation pressures negatively impacted margins | 0     |

## This document serves as an independent evaluation set, utilized both to stress-test the model's predictive accuracy and to automatically extract actionable strategic Pros and Cons. Evaluation Document : NYSE_NKE_2023 (Nike FY 2023 Impact Report) https://github.com/shoebabdullah/kraftwerk/blob/main/NYSE_NKE_2023.pdf

The workflow of this capstone project is split into two core phases. 

First, a multi-model supervised learning architecture is trained and cross-validated on a 3,000-sentence textual dataset (CapstoneDataSet.csv). 

Second, the optimized model is applied during production inference to process unseen layouts within the Nike Fiscal Year 2023 Impact Report. Project success is benchmarked on the pipeline's capability to accurately extract, categorize, and visualize corporate Pros and Cons from this raw impact disclosure document.

## Step 1: Data Preparation

The system:

Loads the dataset
Removes null values
Converts labels to numeric format
Enforces data consistency
Output

Clean dataset ready for machine learning.

## Step 2: Exploratory Data Analysis (EDA)
Class Distribution Analysis : Visualizes the balance between Pros (Label 1) and Cons (Label 0).

<img width="940" height="548" alt="image" src="https://github.com/user-attachments/assets/17a446c1-31f1-4c6f-a84c-7f4c0166f764" />

Text Length Analysis : Calculates Character counts and Word counts.
Generates histograms showing how document lengths vary across sentiment classes.

<img width="923" height="385" alt="image" src="https://github.com/user-attachments/assets/4ecc054c-5a23-49d7-a6ae-0bd8bc5fd0be" />

## Step 3: Feature Engineering and Advanced Exploratory Data Analysis (EDA)

Before feeding the text into machine learning models, the unstructured text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorization.
* Common English stop words were removed.
* The feature space was limited to the top 1,500 most significant text elements to reduce noise.

**TF-IDF Vectorization:** The project converts text into numerical features. This captures the importance of words across documents while reducing noise.
**Top TF-IDF Features:** The system identifies the most influential words across the corpus.

<img width="935" height="472" alt="image" src="https://github.com/user-attachments/assets/da152eed-2061-42fd-a0f7-dc8d98b9a7a2" />

**Word Cloud Analysis:** Separate word clouds are generated for: Positive statements & Negative statements
These visualizations help identify dominant themes in each sentiment category.

<img width="932" height="263" alt="image" src="https://github.com/user-attachments/assets/0be358a2-b649-4aab-aa11-af0e0ae6142d" />

## Step 4: Modeling - GRIDSEARCH HYPERPARAMETER TUNING & CROSS-VALIDATION

**Supervised Machine Learning**
**Models Evaluated**
    Logistic Regression - A linear classification model commonly used for text classification.
    Decision Tree - A rule-based model that learns decision paths from training data.
    Random Forest - An ensemble of decision trees designed to improve prediction accuracy and reduce overfitting.
    Gradient Boosting - A boosting algorithm that sequentially improves prediction performance.

**Hyperparameter Optimization**
Each model undergoes optimization using: GridSearchCV. Parameters are systematically tested to identify the best-performing configuration.

**Cross Validation**
The project applies 3-Fold Cross Validation to measure model stability and generalization performance.

## Step 5: Visualizing best model performance metrics

**Metrics collected:**
    Accuracy
    Precision
    Recall
    F1 Score
<img width="951" height="658" alt="image" src="https://github.com/user-attachments/assets/00e1827f-4b5b-4afa-80cb-417356c2a94b" />


### 2. Unsupervised Pattern Discovery (PCA & K-Means)
To observe natural groupings blindly (without using labels), an unsupervised pipeline was applied:
* **PCA (Principal Component Analysis):** Reduced the 1500-dimensional TF-IDF space down to 2 principal components for visualization.
* **K-Means Clustering:** Segmented the data into exactly 2 natural clusters based on mathematical similarities.

*Insight:* While K-Means cleanly splits the space into two clusters, comparing this to the true labels reveals a dense region of overlap in the center. Because both positive and negative points share common business terminology (e.g., "product interface"), simple word counts blend together. 

*(Note: Added `class_distribution.png`, `pca_kmeans_clusters.png`, and `pca_true_labels.png` from your EDA to the repository).*

## Next Steps & Future Work (This will be completed in final capstone project submission in Module 24)
To improve the tool and fully address the research question, the next phases of the project will focus on:
1. **Contextual Embeddings:** Transitioning from TF-IDF to contextual deep learning embeddings (e.g., BERT, Word2Vec) to better understand word associations and disambiguate overlapping terminology.
2. **Ensemble & Advanced Methods:** Utilizing algorithms like Random Forests, XGBoost, or Deep Neural Networks to capture non-linear relationships that the baseline model may have missed.


## Expanded Model Comparison
To further investigate the research question, I have implemented several advanced supervised learning algorithms. Below is a comparison of their performance:

###              Model  CV Mean Accuracy  Test Accuracy  Precision  F1-Score
### Logistic Regression          0.781309       0.791667   0.800687  0.788494
###       Random Forest          0.772545       0.783333   0.821970  0.769504
###   Gradient Boosting          0.736651       0.743333   0.880208  0.686992
###       Decision Tree          0.744582       0.740000   0.746575  0.736486

*Results show that Logistic Regression performed the best with an accuracy of 78%.*

## Conclusion
The results confirm that machine learning can effectively automate the extraction of "Pros" and "Cons" from unstructured text. Future iterations will focus on transformer-based embeddings to improve semantic understanding.
