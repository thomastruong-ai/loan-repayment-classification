# Loan Repayment Classification

Final project for IBM's *Machine Learning with Python* (ML0101EN). The task: given a historical loan dataset, predict whether a borrower will pay off their loan or go to collection — and compare four classification algorithms on the same problem to see which one the data actually favours.

## The problem

Each record describes a completed loan: principal amount, term length, effective and due dates, plus borrower age, gender, and education level. The target is `loan_status` — **PAIDOFF** or **COLLECTION**.

The dataset is small (~346 training records) and imbalanced toward PAIDOFF, which matters for evaluation: raw accuracy flatters any model that simply predicts the majority class. That's why the comparison below uses Jaccard index, F1-score, and log loss rather than accuracy alone.

## Results

> **[EDIT — fill in from the final evaluation cell of the notebook.]**

| Algorithm | Jaccard | F1-score | LogLoss |
|---|---|---|---|
| K-Nearest Neighbors | | | NA |
| Decision Tree | | | NA |
| Support Vector Machine | | | NA |
| Logistic Regression | | | |

**[EDIT — one or two sentences: which model performed best, by how much, and whether the margin is meaningful given the size of the test set (~54 records). If the models are close, say so — that is the honest reading.]**

## Approach

1. **Exploration** — distribution of loan status by principal, age, gender, and education
2. **Feature engineering** — extracted day-of-week from the effective date and derived a weekend indicator, which separates the classes better than the raw date; converted categorical fields to numeric via one-hot encoding
3. **Preprocessing** — standardized features so distance-based methods (KNN, SVM) aren't dominated by the scale of `Principal`
4. **Modeling** — KNN with *k* selected by validation accuracy across a range of values; decision tree with tuned depth; SVM with an RBF kernel; logistic regression
5. **Evaluation** — all four models scored on a held-out test set using Jaccard, F1, and log loss

## Notes on method

Standardization is fitted on the training data and applied to the test data, not fitted separately on each — otherwise the test set leaks information about its own distribution into the model.

Log loss is only reported for logistic regression, since it requires calibrated probability estimates that the other three models here don't produce by default.

## Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter`

## Running it

```bash
git clone https://github.com/thomastruong-ai/MachineLearning.git
cd MachineLearning
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook ML0101EN-Proj-Loan-py-v1_Final.ipynb
```

The notebook downloads `loan_train.csv` and `loan_test.csv` at runtime; no local data files are required.

---

Coursework project — IBM Machine Learning with Python (ML0101EN), part of the IBM Data Science Professional Certificate.

[Thomas Truong](https://github.com/thomastruong-ai) · PhD, Computer Science
