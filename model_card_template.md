# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
**Model Type:** Random Forest Classifier

**Model Version:** 1.0

**Developer:** Student (Census Income Prediction Project)

**Model Date:** October 2025

**Framework:** Scikit-learn (RandomForestClassifier)

**Hyperparameters:**
- n_estimators: 100
- max_depth: 20
- min_samples_split: 10
- min_samples_leaf: 4
- random_state: 42

The model was trained using 5-fold cross-validation to select the best performing model based on F1 score.

## Intended Use

Predict whether an individual's annual income exceeds $50,000 based on census data attributes.

## Training Data

**Dataset Source:** https://archive.ics.uci.edu/dataset/20/census+income

**Size:** Approximately 32,561 instances

**Features Used:**
- **Continuous Features:** age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- **Categorical Features:** workclass, education, marital-status, occupation, relationship, race, sex, native-country

**Target Variable:** Binary classification - income <=50K or >50K

**Data Preprocessing:**
- Categorical features: One-hot encoding
- Target variable: Label binarization
- All spaces removed from raw data
- 5-fold cross-validation split

## Evaluation Data

**Evaluation Method:** 5-fold cross-validation on the entire dataset

**Data Split:** Each fold contains approximately 20% of the data as test set

The model was evaluated on folds during cross-validation, and the best performing model based on F1 score was selected.

## Metrics

**Metrics Used:**
- **Precision:** Proportion of positive predictions that were correct
- **Recall:** Proportion of actual positives that were correctly identified
- **F1 Score:** Harmonic mean of precision and recall

**Overall Performance (5-Fold Cross-Validation Average):**
- Average Precision: ~0.77 
- Average Recall: ~0.58
- Average F1 Score: ~0.69

**Note:** Exact values are stored in `model/kfold_metrics.csv`

**Performance on Data Slices:**

The model performance was evaluated across different categorical feature values (slices). Detailed results are available in `model/slice_output.txt`. Key observations:

- Performance varies across different demographic groups
- Some occupations and education levels show better prediction accuracy
- Certain slices with smaller sample sizes show more variable performance

## Ethical Considerations

**Bias Concerns:**
- The model may reflect historical biases present in the census data
- Performance disparities exist across different demographic groups (race, sex, native country)
- The model should NOT be used for making decisions that could discriminate against protected classes

**Fairness:**
- Model performance should be evaluated for fairness across sensitive attributes
- Slice-based performance metrics reveal potential disparities
- Further fairness analysis is recommended before any deployment

**Privacy:**
- The training data is publicly available census data
- No individual level privacy concerns for the model itself
- Care should be taken when deploying to avoid inference attacks

## Caveats and Recommendations

**Limitations:**
- Model trained on historical census data may not reflect current economic conditions
- Performance varies significantly across demographic slices
- Binary income threshold (50K) is arbitrary and may not reflect meaningful economic distinctions in all contexts
- Limited feature set may not capture all factors affecting income

**Recommendations:**
- Do NOT use this model for high-stakes decision making (lending, hiring, etc.)
- Conduct thorough fairness audits before any real-world application
- Consider ensemble approaches or more sophisticated models for production use
- Regularly retrain with updated data to maintain relevance
- Implement fairness constraints during training if deploying in sensitive contexts
- Use slice-based evaluation to understand model behavior across different populations
- Consider using the model as a baseline for developing more fair and accurate systems


