# Machine Learning Assignment 2

## 1. Problem Statement
The goal of this project is to implement and evaluate six different classification algorithms to predict Default of Credit Card Clients using the UCI dataset (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). The project involves data preprocessing, model training, evaluation using multiple metrics, and deployment of the final solution as an interactive Streamlit web application.

## 2. Dataset Description
* **Dataset Name:** Default of Credit Card Clients dataset
* **Source:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
* **Description:** The dataset consists of 15000 instances and 23 features. It includes attributes such as LIMIT_BAL, EDUCATION etc to classify samples into default yes/no categories.

## 3. Models Used & Evaluation Metrics
The following six machine learning models were implemented and evaluated. The performance metrics for each model are summarized below:

|               Model | Accuracy | AUC Score | Precision | Recall | F1-Score |    MCC |
|--------------------:|---------:|----------:|----------:|-------:|---------:|-------:|
| Logistic_Regression |   0.8117 |    0.7113 |    0.6907 | 0.2489 |   0.3659 | 0.3341 |
|       Decision_Tree |   0.8187 |    0.7398 |    0.6504 | 0.3664 |   0.4688 | 0.3917 |
|                 KNN |   0.7923 |    0.6883 |    0.5385 | 0.3420 |   0.4183 | 0.3109 |
|         Naive_Bayes |   0.3900 |    0.6670 |    0.2453 | 0.8641 |   0.3822 | 0.1193 |
|       Random_Forest |   0.8033 |    0.7271 |    0.5881 | 0.3313 |   0.4238 | 0.3352 |
|             XGBoost |   0.8073 |    0.7363 |    0.5932 | 0.3740 |   0.4588 | 0.3626 |

## 4. Observations on Model Performance
Below are the specific observations regarding how each model performed on the chosen dataset:

| ML Model Name            | Observation about model performance                                                                                                                                                                               |   |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| Logistic Regression      | Strong Baseline: Achieved a solid 81.1% accuracy. While reliable for non-defaults, it had the lowest recall for defaults (0.25), indicating it struggles to identify risky clients compared to non-linear models. |   |
| Decision Tree            | Top Performer: Surprising leader in this run with the highest Accuracy (0.818) and AUC (0.740). It achieved a better balance for the minority class (Recall 0.37) than the linear baseline.                       |   |
| KNN Classifier           | Moderate: Performance was lower than tree-based models (AUC 0.688). This suggests that "closeness" in the feature space is not the strongest indicator of credit risk in this specific 23-feature dataset.        |   |
| Naive Bayes              | Poor Fit: The model failed significantly on accuracy (0.39) because it over-predicted defaults. While it had high recall (0.86), the extremely low precision makes it impractical for this use case.              |   |
| Random Forest (Ensemble) | Under-utilized: Performed slightly worse than the single Decision Tree. This is likely because it was restricted to only 10 estimators, which wasn't enough to leverage the power of the ensemble.                |   |
| XGBoost (Ensemble)       | Robust & Competitive: Delivered a strong AUC of 0.736. It matched the Decision Tree's recall (0.37) for defaults, showing it is effectively handling the complex relationships in the data.                       |   |

## 5. Project Links
* **Live App:** https://2025aa05327mlassignment2.streamlit.app/
* **GitHub Repository:** https://github.com/anirbansen2709/2025AA05327_ML_Assignment2

## 6. How to Run Locally
1.  Clone the repository:
    ```bash
    git clone https://github.com/anirbansen2709/2025AA05327_ML_Assignment2.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
