# Titanic — Pipeline, Features, and Model Selection

This repository contains my implementation of **Lab 5: Titanic Machine Learning**, covering data exploration, data cleaning, leakage-safe preprocessing, feature engineering, model comparison, hyperparameter tuning, and final evaluation.

The project is divided into two days:

* **Day 1:** Data exploration, data quality analysis, cleaning decisions, and train/test split.
* **Day 2:** Preprocessing pipelines, cross-validation, feature engineering, model comparison, hyperparameter tuning, and final test evaluation.

The Day 2 work continues directly from the train/test split created on Day 1. The objective was to build a reliable machine learning workflow where preprocessing and model selection are performed without leaking information from the validation or test data.

---

## Project Structure

```text
MSAI_py_lab5/
│
├── data.py
├── pipeline.py
├── split.joblib
│
├── notebooks/
│   ├── day1_exploration_starter.ipynb
│   └── day2_pipeline_modeling_starter.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses the Titanic passenger dataset obtained through OpenML.

The dataset contains passenger information such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Embarkation port
* Cabin information
* Passenger name
* Ticket information
* Other outcome-related fields

The target variable is:

```text
Survived
```

where:

* `0` = did not survive
* `1` = survived

---

# Day 1 — Data Exploration and Cleaning

The first stage focused on understanding the dataset before building models.

### Missing values

The main missing-value issues were found in:

* `body`
* `Cabin`
* `boat`
* `home.dest`
* `Age`

There were also a very small number of missing values in `Embarked` and `Fare`.

### Cleaning decisions

The following decisions were made for the baseline model:

| Feature     | Decision                 | Reason                                                                                        |
| ----------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| `Age`       | Keep and impute later    | Useful passenger information with moderate missingness                                        |
| `Fare`      | Keep and impute later    | Potentially useful for passenger socioeconomic status                                         |
| `Embarked`  | Keep and impute later    | Useful categorical feature                                                                    |
| `Cabin`     | Convert to `Cabin_known` | Preserve whether cabin information was available without using the high-cardinality raw value |
| `boat`      | Drop                     | Outcome-related information and target leakage                                                |
| `body`      | Drop                     | Strongly related to the outcome and highly incomplete                                         |
| `home.dest` | Drop                     | High missingness and high-cardinality information                                             |
| `Name`      | Drop from baseline       | High-cardinality text feature                                                                 |
| `Ticket`    | Drop                     | High-cardinality identifier-like feature                                                      |
| `Survived`  | Target                   | Not included in the input features                                                            |

The resulting baseline feature set contains:

```text
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
Cabin_known
```

`Pclass` is treated as a categorical variable even though it is numerically encoded because passenger classes represent discrete categories rather than a continuous measurement.

---

# Train/Test Split

The dataset was split before fitting any preprocessing transformations.

The split used:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)
```

The resulting sizes were:

```text
X_train: (1047, 8)
X_test : (262, 8)
y_train: (1047,)
y_test : (262,)
```

Stratification was used to preserve approximately the same target-class proportions in the training and test sets.

The split was saved as:

```text
split.joblib
```

so that Day 2 could continue from the same split rather than creating a new one.

---

# Day 2 — Pipeline and Model Selection

The main goal of Day 2 was to create a **leakage-safe machine learning workflow**.

The workflow was:

```text
Training Data
     │
     ▼
ColumnTransformer
     │
     ├── Numerical features
     │      ├── Median imputation
     │      └── Standard scaling
     │
     └── Categorical features
            ├── Most-frequent imputation
            └── One-hot encoding
     │
     ▼
Machine Learning Model
     │
     ▼
Cross-validation
     │
     ▼
Model selection
     │
     ▼
Hyperparameter tuning
     │
     ▼
Final test evaluation
```

---

## Leakage Demonstration

Before constructing the pipeline, preprocessing was deliberately fitted once using the complete dataset and once using only the training data.

The learned means were different:

```text
                 Full Dataset    Training Dataset
Age              29.881135       29.604316
SibSp             0.498854        0.484241
Parch             0.385027        0.385864
Fare             33.295479       32.166838
Cabin_known       0.225363        0.214900
```

This demonstrates that fitting preprocessing using the full dataset allows information from the test set to influence the transformation.

Therefore, preprocessing was placed inside a `Pipeline`, ensuring that fitted preprocessing steps are learned only from the appropriate training data during cross-validation.

---

# Preprocessing

A `ColumnTransformer` was used to apply different preprocessing to numerical and categorical variables.

### Numerical features

The numerical features were:

```text
Age
SibSp
Parch
Fare
Cabin_known
```

They were processed using:

1. Median imputation
2. Standard scaling

### Categorical features

The categorical features were:

```text
Pclass
Sex
Embarked
```

They were processed using:

1. Most-frequent imputation
2. One-hot encoding

The encoder uses:

```python
handle_unknown="ignore"
```

so that an unseen category during prediction does not cause the pipeline to fail.

The preprocessing and model are kept together in a single scikit-learn `Pipeline`.

---

# Cross-Validated Baseline

Logistic Regression was selected as the baseline model because it provides a simple reference point for evaluating more complex models and engineered features.

The evaluation used:

```text
5-fold Stratified Cross-Validation
Metric: F1-score
```

Baseline results:

```text
F1 scores:
[0.7317, 0.6667, 0.7195, 0.7020, 0.7123]

Mean F1: 0.7064
Standard deviation: 0.0221
```

The mean F1 represents the average performance across folds, while the standard deviation shows how much the result varies between folds.

---

# Feature Engineering

A feature-engineering hypothesis was stated before constructing the features.

### Hypothesis

> I hypothesize that family structure may contain useful information about passenger survival. Passengers travelling with family members may have different survival outcomes from passengers travelling alone.

Two features were tested:

* `FamilySize`
* `IsAlone`

The features were evaluated using the same cross-validation procedure as the baseline.

### Results

| Version              |  Mean F1 |   Std F1 |
| -------------------- | -------: | -------: |
| Baseline             | 0.706440 | 0.022117 |
| FamilySize           | 0.706440 | 0.022117 |
| IsAlone              | 0.702682 | 0.022560 |
| FamilySize + IsAlone | 0.702682 | 0.022560 |

`FamilySize` produced essentially the same result as the baseline, while `IsAlone` and the combination of both features produced slightly lower scores.

Therefore, the original hypothesis was **not clearly supported** by the cross-validation results.

The negative result was retained and reported rather than selecting features simply because they were expected to improve performance.

---

# Model Comparison

Three different model types were evaluated using the same preprocessing, training data, cross-validation strategy, and F1-score:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

### Results

| Model               |  Mean F1 |   Std F1 |
| ------------------- | -------: | -------: |
| SVM                 | 0.727745 | 0.010555 |
| Random Forest       | 0.714145 | 0.015266 |
| Logistic Regression | 0.702682 | 0.022560 |

SVM achieved the highest mean F1 score.

However, the differences between the models are relatively small compared with the fold-to-fold variation. Therefore, SVM was treated as the strongest candidate for further investigation rather than assuming that it was dramatically better than the other models.

---

# Hyperparameter Tuning

Since SVM achieved the strongest cross-validated result, it was selected for hyperparameter tuning.

`GridSearchCV` was performed using the training data only.

The parameters searched were:

```text
C
gamma
kernel
```

The best configuration was:

```text
kernel = rbf
C      = 1
gamma  = 0.1
```

The best cross-validation F1 score was:

```text
0.727997
```

This was only a very small improvement over the untuned SVM cross-validation F1 score of 0.727745, showing that hyperparameter tuning had little effect on the cross-validated performance in this experiment.

The test set was not used during hyperparameter selection.


---

# Final Test Evaluation

After model selection and tuning were completed, the held-out test set was evaluated exactly once.

The final test result was:

```text
Final Test F1: 0.7812
```

The classification report was:

```text
              precision    recall  f1-score   support

           0       0.85      0.90      0.87       162
           1       0.82      0.75      0.78       100

    accuracy                           0.84       262
   macro avg       0.83      0.82      0.83       262
weighted avg       0.84      0.84      0.84       262
```

The test set was deliberately kept separate from preprocessing fitting, model comparison, and hyperparameter tuning.

---

# Key Findings

### 1. Leakage prevention matters

Fitting preprocessing on the full dataset produced different learned statistics compared with fitting only on the training data. Keeping preprocessing inside a Pipeline prevents this form of leakage.

### 2. Family features did not improve the baseline

`FamilySize` performed almost identically to the baseline, while `IsAlone` slightly reduced F1. Therefore, the tested family-based feature hypothesis was not supported.

### 3. SVM performed best in cross-validation

SVM achieved the highest mean F1 score among the three tested models:

```text
SVM                  0.7277
Random Forest        0.7141
Logistic Regression  0.7027
```

The difference was not large enough to claim a dramatic performance gap.

### 4. Hyperparameter tuning was performed without test-set leakage

The SVM was tuned using only the training data with `GridSearchCV`.

### 5. Final test performance

The tuned SVM achieved:

```text
Test F1 = 0.7812
```

on the held-out test set.

---



# Technologies Used

* Python 3.11
* pandas
* NumPy
* scikit-learn
* joblib
* Jupyter Notebook

Main scikit-learn components used:

```text
train_test_split
StratifiedKFold
ColumnTransformer
Pipeline
SimpleImputer
StandardScaler
OneHotEncoder
LogisticRegression
RandomForestClassifier
SVC
cross_val_score
GridSearchCV
f1_score
classification_report
```

---

# How to Run

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks:

```text
notebooks/day1_exploration_starter.ipynb
notebooks/day2_pipeline_modeling_starter.ipynb
```

Run the notebooks from top to bottom after restarting the kernel.

Day 2 uses the `split.joblib` file generated from Day 1, so the Day 1 split should be created before running Day 2.

---

# Reproducibility

Random seeds were specified where appropriate, including:

```text
Train/test split: random_state=42
Cross-validation: random_state=42
Random Forest: random_state=42
```

The preprocessing steps are part of the model pipeline so that they are refitted correctly within the cross-validation process.

---

# Conclusion

This project demonstrates a complete baseline machine learning workflow for the Titanic classification problem.

The main lessons from the experiment were that preprocessing must be kept inside the pipeline to avoid leakage, engineered features should be tested rather than assumed to be useful, model comparisons should include both mean performance and variability, and the test set should remain untouched until the final evaluation.

The final tuned SVM achieved a test F1 score of **0.7812**.

