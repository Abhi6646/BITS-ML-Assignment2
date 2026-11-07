# <Your project title>

## a. Problem statement
<The goal of this project is to predict a person's obesity level from their eating habits, physical condition and basic body measurements. This is a multi-class classification problem with 7 classes, ranging from Insufficient Weight to Obesity Type III. Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the same dataset, compared on six metrics, and exposed through an interactive Streamlit app.>

## b. Dataset description  [1 mark]
- **Source:** <[Kaggle / UCI link](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)>
- **Instances:** <2111>   
-**Features:** <16 input features with one target column>   
-**Target:** <n'NObeyesdad', with 7 classes: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. The classes are fairly balanced.>
- **Feature types:** <a mix of numeric features (Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE) and categorical features (Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS).>
- **Preprocessing:** <missing numeric values are filled with the median and scaled with StandardScaler. Categorical features are imputed with the most frequent value and one-hot encoded. All preprocessing is part of a scikit-learn Pipeline saved together with each model, so the Streamlit app can take a raw CSV directly.>

## c. GitHub repository  [1 mark]
<https://github.com/Abhi6646/BITS-ML-Assignment2>

Live app: <https://<your-app>.streamlit.app>

## d. Models used  [5 marks]

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.8747 | 0.9837 |      0.8729 |   0.8747 | 0.8731 | 0.8539 |
| Decision Tree          |     0.9149 | 0.9586 |      0.9173 |   0.9149 | 0.9154 | 0.9008 |
| KNN                    |     0.8109 | 0.9718 |      0.8124 |   0.8109 | 0.7986 | 0.7826 |
| Naive Bayes            |     0.5083 | 0.8382 |      0.5275 |   0.5083 | 0.4607 | 0.4471 |
| Support Vector Machine |     0.9243 | 0.9931 |      0.9264 |   0.9243 | 0.9248 | 0.9118 |

Because the problem is multi-class, Precision, Recall and F1 are weighted averages across classes, and AUC is one-vs-rest weighted.

Hyperparameters used: Logistic Regression ('max_iter=2000'), Decision Tree ('max_depth=8'), KNN ('k=7'), Gaussian Naive Bayes (defaults), SVM (RBF kernel, 'probability=True').

### Observations  [3 marks]

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Reached 0.8747 accuracy and a very high AUC of 0.9837, so it ranks the classes well, but it is a linear model and the boundaries between neighbouring obesity levels are not perfectly linear, which limits accuracy compared with the tree and the SVM. |
| Decision Tree | Reached 0.9149 accuracy and 0.9008 MCC, the second best result. It can learn threshold-style splits on features like Weight and Height. Its AUC (0.9586) is lower than the other good models because a tree gives coarse probability estimates. |
| KNN | Scored 0.8109 accuracy, and its F1 (0.7986) is below its accuracy, which suggests uneven performance across classes. Distance-based methods suffer when one-hot encoded features add many dimensions, even after scaling. |
| Naive Bayes | The weakest model with 0.5083 accuracy and 0.4471 MCC. It assumes features are independent and Gaussian, which does not hold here: Height and Weight are strongly related and the one-hot columns are binary. |
| Support Vector Machine | Best on every metric: 0.9243 accuracy, 0.9931 AUC and 0.9118 MCC. The RBF kernel handles the non-linear class boundaries well. |
| **Overall winner for this dataset** | **Support Vector Machine**, with the highest accuracy, AUC, F1 and MCC. The Decision Tree is a close second and is easier to interpret. |



## Repository structure
```
app.py               Streamlit application
eval_utils.py        shared metric calculation
train_models.py      training and evaluation script
requirements.txt     dependencies
test_data.csv        held-out test data for the app
model/               saved pipelines (.joblib), meta.json, metrics_summary.csv
```

## How to run
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```