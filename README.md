**Professional Certificate in Machine Learning and Artificial Intelligence**
<br>**Practical Application 3**<br /> 
<br>**Required Assignment 17.1: Comparing Classifiers**<br /> 
<br>* **Link to access the Jupyter Notebook** [Assignment Jupyter Code]https://github.com/shoebabdullah/kraftwerk/blob/main/Practical%20Application%203%20Compairing%20Classifiers.ipynb<br /> 
<br># Bank Marketing Campaign: Comparing Classifiers<br /> 
<br>## Project Overview<br /> 
<br>The objective of this project is to develop a predictive model that accurately identifies the likelihood of a client subscribing to a bank term deposit. Using a dataset from a Portuguese banking institution (UCI Machine Learning Repository), we compare the performance of four primary classifiers:<br /> 
<br>* **K-Nearest Neighbors (KNN)**<br /> 
<br>* **Logistic Regression**<br /> 
<br>* **Decision Trees**<br /> 
<br>* **Support Vector Machines (SVM)**<br /> 
<br>## 1. Understanding the Data<br /> 
<br>The dataset represents **17 marketing campaigns** conducted between May 2008 and November 2010. It includes:
<br>* **Bank client data**: Age, job, marital status, education, etc.<br /> 
<br>* **Communication details**: Contact type, day of the week, and month.<br /> 
<br>* **Socio-economic indicators**: Employment variation rate, consumer price index, and euribor 3-month rate.<br /> 

<br>## 2. Data Preparation & Engineering<br /> 
<br>To ensure a realistic and efficient model, the following preprocessing steps were taken:<br /> 
<br>* **Handling Missing Values**: Categorical 'unknown' values were treated as missing data and replaced with `NaN`.<br /> 
<br>* **Data Leakage Prevention**: The `duration` feature was dropped because it is only known after a call is performed and is highly correlated with the target variable.<br /> 
<br>* **Feature Encoding**: <br /> 
<br>    * The target variable `y` was coerced into a binary numeric format (1 for 'yes', 0 for 'no').<br /> 
<br>    * Categorical features (job, marital, education, etc.) were transformed using **One-Hot Encoding** to create binary columns.<br /> 
<br>* **Data Optimization**: Object columns were converted to optimized `category` types to improve memory efficiency.<br /> 

<br>## 3. Business Objective<br /> 
<br>The goal is to move beyond mass-marketing and focus on "high-potential" clients. Key benefits include:<br /> 
<br>* **Optimized Efficiency**: Focusing resources on likely converts.<br /> 
<br>* **Reduced Costs**: Minimizing unnecessary contact attempts.<br /> 
<br>* **Enhanced Customer Experience**: Reducing intrusive calls to uninterested clients.<br /> 

<br>## 4. Modeling Approach<br /> 
<br>### Baseline Performance<br /> 
<br>A baseline was established to determine the minimum accuracy the models must beat, typically representing the majority class (non-subscribers).<br /> 

<br>### Train/Test Split<br /> 
<br>The data was split into a **Training set (80%)** and a **Testing set (20%)** to evaluate how well the models generalize to unseen data.<br /> 
<br>* **Training set shape**: (32950, 53)<br /> 
<br>* **Testing set shape**: (8238, 53)<br /> 

<br>## 5. Preliminary Findings<br /> 
<br>Initial analysis of the Support Vector Machine (SVM) and other models suggests:<br /> 
<br>* **Tuning**: Accuracy varies significantly based on kernel choice and complexity ($C$ values).<br /> 
<br>* **Recall Challenges**: While models are precise in identifying non-subscribers, they often miss a portion of actual subscribers (lower recall), which is a critical area for improvement in marketing campaigns.<br /> 

<br>## How to Run<br /> 
<br>1. Ensure you have `pandas`, `numpy`, and `scikit-learn` installed.<br /> 
<br>2. Place the `bank-additional-full.csv` file in the appropriate directory.<br /> 
<br>3. Run the cells in `Practical Application 3 Compairing Classifiers.ipynb` sequentially.<br /> 
<br>* **#Key Observations:**<br /> 
<br>* **#Baseline Proximity:** Most models achieve a test accuracy around 88.6%, which is the same as the baseline (predicting "no" for everyone). This indicates that the basic bank demographic features alone provide limited predictive power for identifying subscribers.<br /> 

<br>* **#Training Speed:**<br /> 

<br>* **#KNN** and **Logistic Regression** were the fastest to train, making them highly efficient for initial assessments.<br /> 

<br>* **#SVM** was significantly slower, taking over 80 seconds to train, which is a major drawback as the dataset scales.<br /> 

<br>* **#Overfitting:** The Decision Tree showed signs of overfitting, achieving the highest training accuracy (91.66%) but the lowest test accuracy (86.10%). This suggests the tree grew too complex and captured noise in the training data.<br /> 

<br>* **#Model Stability:** Logistic Regression and SVM provided the most stable results between training and testing sets, although they largely defaulted to predicting the majority class.<br /> 

<br>* **Model**  **Train Time (s)**  **Train Accuracy**  **Test Accuracy**<br /> 
<br>**0**  Logistic Regression          0.0320          0.8876         **0.8865**<br />
<br>**1**                  KNN          0.0055          0.8897         0.8736<br />
<br>**2**        Decision Tree          0.1163        **0.9166**       0.8610<br />
<br>**3**                  SVM        **24.9186**      0.8878         0.8867<br />
