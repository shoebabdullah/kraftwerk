# Required Capstone Project Module 24.1 - Final Report : Automated Extraction of Strategic Advantages (Pros) and Operational Risks (Cons) from Business Documents Using Machine Learning.
## Project Overview : Reading through long business documents to find the "good" and "bad" points takes too much time and often leads to people missing important details. This project builds a smart tool that automatically scans these documents to pull out a balanced list of Pros and Cons, helping people make faster, unbiased decisions.

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

## Key Findings & Analysis :

**Top Performer (Accuracy & F1-Score): Logistic Regression** achieved the highest overall performance in multiple key categories. It leads with a **Test Accuracy of 0.791667**, a CV Mean Accuracy of 0.781309, and the highest F1-Score of 0.788494.

**Strongest Precision: Gradient Boosting** stands out significantly for its **Precision (0.880208)**, meaning it has the lowest rate of false positives among all the models. However, its overall F1-Score (0.686992) is the lowest, indicating a potential trade-off with its recall.

**The Baseline Tree Models: Random Forest** performs solidly in second place for accuracy, while the single **Decision Tree** trails slightly behind, though it maintains a more balanced precision-to-F1 ratio than Gradient Boosting.

<img width="750" height="270" alt="image" src="https://github.com/user-attachments/assets/36fa6930-9097-4729-b20b-abe3649f61e0" />


* **Model	            | CV Mean Accuracy |	Test Accuracy |	Precision |	F1-Score**
* Logistic Regression	|   0.781309	   |    0.791667	  |  0.800687 |	0.788494
* Random Forest	            0.772545	        0.783333	    0.821970	0.769504
* Gradient Boosting	        0.736651	        0.743333	    0.880208	0.686992
* Decision Tree	            0.744582	        0.740000	    0.746575	0.736486

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
*    **Logistic Regression** - A linear classification model commonly used for text classification.
*    **Decision Tree** - A rule-based model that learns decision paths from training data.
*    **Random Forest** - An ensemble of decision trees designed to improve prediction accuracy and reduce overfitting.
*    **Gradient Boosting** - A boosting algorithm that sequentially improves prediction performance.

**Hyperparameter Optimization**
Each model undergoes optimization using: GridSearchCV. Parameters are systematically tested to identify the best-performing configuration.

**Cross Validation**
The project applies 3-Fold Cross Validation to measure model stability and generalization performance.

## Step 5: Visualizing Best Model Performance Metrics
To further investigate the research question, I have implemented several advanced supervised learning algorithms. 

**Metrics collected:**
    Accuracy
    Precision
    Recall
    F1 Score
<img width="951" height="658" alt="image" src="https://github.com/user-attachments/assets/00e1827f-4b5b-4afa-80cb-417356c2a94b" />

## Step 6: Executing PCA and Unsupervised K-Means Clustering

**Unsupervised Learning**
**Principal Component Analysis (PCA):** The high-dimensional TF-IDF feature space is reduced to two dimensions.
Purpose: Visualization, Pattern discovery and Cluster separation analysis.

**K-Means Clustering:** The system applies KMeans(n_clusters=2) to discover natural groupings in the text data.

## Step 7: Comparing Clustering VS. Ground Truth Labels
**Cluster Comparison**
The project compares, Machine-discovered clusters and Human-provided sentiment labels.

<img width="973" height="348" alt="image" src="https://github.com/user-attachments/assets/d24ee227-0a08-4402-827d-cf8d5134f8c3" />

## Step 8: PDF Document Analysis Module
A unique component of this capstone is the external document analysis engine.
Expected PDF location: NYSE_NKE_2023.pdf

The system:

* Reads PDF pages
* Extracts text
* Cleans formatting
* Splits content into sentences
* Filters irrelevant statements

If no PDF exists, a fallback financial document simulation is used.

**Strategic Insight Extraction**
The project performs rule-based sentiment scoring on extracted PDF sentences.

**Positive Indicators**

Examples: Growth, Expansion, Profit, Efficiency, Innovation, 

**Negative Indicators**

Examples: Risk, Competition, Inflation, Volatility, Decline

**Strategic Reporting**

The engine automatically generates:

**Top Strategic Pros**

Examples:

* Growth opportunities
* Operational efficiencies
* Market expansion initiatives

**Top Strategic Cons**

Examples:

* Inflationary pressures
* Competitive threats
* Margin compression risks

This produces an executive-style summary that can assist decision-makers during document reviews.

## Step 9: Strategic Word Plot Visualization
Additional word clouds are created from extracted PDF insights.

These visualizations highlight:

**Strength Themes** 

* Growth
* Expansion
* Optimization
* Adoption

**Risk Themes**

* Competition
* Inflation
* Volatility
* Expenses
  
<img width="953" height="349" alt="image" src="https://github.com/user-attachments/assets/a2de1ef8-13fe-4f1f-8920-a9f3cf8e70c1" />

**======================= DOCUMENT SENTIMENT SUMMARY REPORT =======================**

**--- TOP STRATEGIC PROS (ADVANTAGES & STRENGTHS) ---**
* 01. Racial and Ethnic Minorities 30 Engagement and Inclusion 36 Engaging the Next Generation 41 Pay and Benefits 49 Career Growth and Development 58 Supplier 66 Code of Conduct INTRODUCTION *If you have a body, you are an athlete.
* 02. In FY23, we provided teammates in Canada and India with improved fertility and family planning support – and will continue to evaluate opportunities to enhance these benefits for other teammates outside of the U.S.
* 03. Eighty-five percent of the employees who used this benefit said it resulted in an improved treatment plan for their healthcare needs.
* 04. Over time, we have seen that improvement in the well-being of workers also leads to improvements in supplier facility performance through decreased turnover and increased productivity.
* 05. After implementing the action plan, facility management saw increased performance and better workforce retention in the area targeted for improvement.
* 06. FY22), expansion of regional service centers that are closer to customers, increased order processing speed and our “no rush” shipping option for digital orders in North America.
* 07. Our ongoing energy efficiency work also helped contribute to Scope 1 emissions reductions, even as we see weather-related increases in energy use.
* 08. While actual emissions in HQs and key offices increased, NIKE’s energy efficiency increased as well.
* 09. This progress is primarily attributed to the continued growth of renewable energy in manufacturing, the reduction of inbound air freight, and the decrease in air freight for digital orders with the expansion of regional service centers and “no rush” shipping options.
* 10. The increase in emissions versus the baseline is primarily due to business growth, continued popularity of higher-carbon- intensity materials in footwear (such as leather), and the rising emissions intensity of the electricity grid in primary manufacturing regions (particularly Vietnam).

**--- TOP STRATEGIC CONS (RISKS & HEADWINDS) ---**
* 01. The committee oversees both the risks and opportunities associated with NIKE’s three purpose pillars: People, Planet and Play.
* 02. For each target our report provides context on our goal, approach and challenges, the initiatives underway to reach the target and updates on the results of our efforts for the fiscal year.
* 03. Those statements, estimates and projections are not guarantees of future results or performance and are subject to certain known and unknown risks and uncertainties that are difficult to predict, are often beyond our control and could cause actual results to differ materially.
* 04. These risks and uncertainties are further detailed in our reports filed with the U.S.
* 05. We also continued to support our employees through challenges related to the COVID-19 pandemic.
* 06. IMPACT REPORT 56 INTRO TA RGETS P EOPLE P LANET P LAY A PPROACH A PPENDIX PUBLIC AWARENESS However, weaving this work into NIKE’s fabric has its challenges, especially internationally.
* 07. 2 D evelop Workers: Focusing on the development of individual competencies and organizational capabilities in risk- appropriate areas.
* 08. We work closely with Pilz, an internationally recognized safety expert, on a program to provide advanced machine safety training and certification for suppliers who operate high-risk machinery.
* 09. 39 W e identify high-risk suppliers using the American Industrial Hygiene Association’s qualitative and quantitative risk assessment and priority setting guidelines.
* 10. In FY23, we developed plans for additional training of high-risk suppliers 39 that aim to be completed in FY24.
 
**Key Learning Outcomes**

This project demonstrates proficiency in:

* Natural Language Processing (NLP)
* Feature Engineering
* Text Classification
* Machine Learning Model Selection
* Hyperparameter Optimization
* Cross Validation
* Unsupervised Learning
* Data Visualization
* PDF Text Mining
* Business Intelligence Analytics

**Future Enhancements**

Potential improvements include:

* Deep Learning models (LSTM, GRU, Transformer architectures)
* BERT-based sentiment classification
* Named Entity Recognition (NER)
* Topic Modeling (LDA)
* Real-time financial news analysis
* Interactive dashboard deployment using Streamlit or Power BI
* Multi-class sentiment classification
* Explainable AI (SHAP and LIME)

## Conclusion
This capstone project presents a comprehensive NLP-driven analytics platform capable of transforming unstructured business text into meaningful insights. By combining supervised learning, unsupervised learning, document parsing, and visualization techniques, the system provides a practical framework for automated sentiment analysis and strategic intelligence extraction from financial and corporate documents.
