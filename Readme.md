# <Your project title>

## a. Problem statement
<2-4 sentences: what are you predicting, why it matters, binary/multi-class.>

## b. Dataset description  [1 mark]
- **Source:** <Kaggle / UCI link>
- **Instances:** <n>   **Features:** <n>   **Target:** <name, classes, class balance>
- **Feature types:** <numeric / categorical, examples>
- **Preprocessing:** <missing values, scaling, encoding, train/test split ratio, random_state>

## c. GitHub repository  [1 mark]
<https://github.com/<user>/<repo>>

Live app: <https://<your-app>.streamlit.app>

## d. Models used  [5 marks]

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.8747 | 0.9837 |      0.8729 |   0.8747 | 0.8731 | 0.8539 |
| Decision Tree          |     0.9149 | 0.9586 |      0.9173 |   0.9149 | 0.9154 | 0.9008 |
| KNN                    |     0.8109 | 0.9718 |      0.8124 |   0.8109 | 0.7986 | 0.7826 |
| Naive Bayes            |     0.5083 | 0.8382 |      0.5275 |   0.5083 | 0.4607 | 0.4471 |
| Support Vector Machine |     0.9243 | 0.9931 |      0.9264 |   0.9243 | 0.9248 | 0.9118 |

### Observations  [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner for this dataset** | |

## Repository structure
```
app.py  eval_utils.py  train_models.py  requirements.txt  README.md  test_data.csv  model/
```

## How to run
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```