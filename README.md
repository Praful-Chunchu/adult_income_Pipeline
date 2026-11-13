# Adult Income ML Pipeline

**Praful Chunchu** | AIPI 510 | Week 12 Interactive

Predicts whether a person earns more than $50K/year from 1994 U.S. Census data using a scikit-learn pipeline and logistic regression.

## Results (test set, evaluated once)

- **Test accuracy = 85.16%**
- Baseline (always predict `<=50K`) = 76.06%
- `>50K` class: precision 0.74, recall 0.59, F1 0.66

Validation accuracy was 85.91% and train accuracy was 85.25%, so the model is not overfitting.

## Data

- Source: OpenML `adult`, version 2, loaded with `sklearn.datasets.fetch_openml`
- 48,842 rows; target is `class` (76% `<=50K`, 24% `>50K`)
- Dropped `fnlwgt` (census sampling weight, not a personal attribute) and `education` (maps 1:1 to `education-num`, which keeps the order)

## Pipeline

1. Split 80/10/10 into train/validation/test, stratified on `class`, `random_state=42`
2. Numeric columns: median imputer, then StandardScaler
3. Categorical columns: constant `"Missing"` imputer (missing becomes its own category), then OneHotEncoder (`handle_unknown="ignore"`)
4. Model: LogisticRegression
5. All preprocessing is inside the Pipeline and fit on train only; validation was used for checking the model; test was used once at the end

## How to run

```
pip install -r requirements.txt
jupyter notebook ML_pipeline.ipynb
```

Then run all cells. The fitted pipeline is saved as `adult_income_pipeline.joblib`.

## Notes

`sex`, `race` and `relationship` are sensitive attributes. This model was not evaluated for fairness.