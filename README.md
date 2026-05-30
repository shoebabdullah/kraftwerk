# Required Capstone Project Module 24.1 - Final Report : Automated Extraction of Pros and Cons from Business Documents
## Overview
## Reading through long business documents to find the "good" and "bad" points takes too much time and often leads to people missing important details. This project builds a smart tool that automatically scans these documents to pull out a balanced list of Pros and Cons, helping people make faster, unbiased decisions.

* **Pros (The "Upside"):** Reasons to say "Yes" (benefits, gains, opportunities).
* **Cons (The "Downside"):** Reasons to say "No" (risks, costs, potential problems).

## Research Question
To what extent can a machine learning model be trained using unsupervised clustering and supervised ensemble methods to accurately extract and categorize "Pros" and "Cons" from unstructured business documents to improve decision-making efficiency?

## Business Problem

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

## This capstone project develops an end-to-end Natural Language Processing (NLP) and Machine Learning pipeline that analyzes business and financial text to identify positive and negative sentiment, compare machine learning models, discover hidden text patterns through clustering, and generate strategic insights from external PDF documents.

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

## Project Architecture
<img width="348" height="722" alt="image" src="https://github.com/user-attachments/assets/c21e483c-bb4a-4553-95ab-ba07ddc70bc4" />

## Technologies Used
* **Programming Language :** Python
* **Data Processing :** NumPy, Pandas
* **Machine Learning :** Scikit-Learn
*          Model Selection :  train_test_split, GridSearchCV, cross_val_score
*          Linear Model    :  LogisticRegression
*          Tree            :  DecisionTreeClassifier
*          Ensemble        :  RandomForestClassifier, GradientBoostingClassifier
*          Decomposition   :  PCA
*          Cluster         :  KMeans
*          Metrics         :  Accuracy_score, Precision_score, Recall_score, F1_score
* **Natural Language Processing :** TF-IDF Vectorization
* **Data Visualization :** Matplotlib, Seaborn, WordCloud
* **Document Processing :** PyPDF

## Dataset : CapstoneDataSet.csv
The dataset consists of **3,000 unstructured text snippets** from business documents and reviews, labeled as either Pros (1) or Cons (0).
* **Cleaned Data:** 2,996 records (4 rows with corrupted/missing labels were removed).
* **Class Distribution:** The dataset is perfectly balanced with a ~50/50 split between Pros and Cons, mitigating class imbalance bias.
  
## Text preprocessing and feature engineering

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
