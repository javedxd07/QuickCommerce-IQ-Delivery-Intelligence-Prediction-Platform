# Random Forest Model Diagnostics

## Scope and method

This is a read-only diagnostic of the saved `RandomForestRegressor` and
`RandomForestClassifier` pipelines. No estimator was fitted, no threshold was
changed, and no existing model, notebook, evaluation result, or prediction
artifact was written.

The pipelines were loaded from `models/` and evaluated on `data/model_ready_data.csv`
using the exact chronological partitions documented in `models/model_metadata.json`
and created in `notebooks/03_Machine_Learning.ipynb`:

| Partition | Rows | Date range |
|---|---:|---|
| Train | 30,739 | 2022-02-11 to 2022-03-23 |
| Validation | 7,525 | 2022-03-24 to 2022-03-30 |
| Test | 7,329 | 2022-03-31 to 2022-04-06 |

The recomputed validation and test metrics matched the saved JSON results. The
test set remains an evaluation set; its results were not used to select or tune
either model. For threshold-dependent classification metrics, the saved
validation-selected threshold of **0.29** and the default **0.50** are both
reported. Confusion matrices use `[[TN, FP], [FN, TP]]`.

## 1. Regression: RandomForestRegressor

### Performance by partition

| Partition | MAE (minutes) | RMSE (minutes) | R² |
|---|---:|---:|---:|
| Train | 3.3099 | 4.2468 | 0.7951 |
| Validation | 6.2519 | 7.8474 | 0.2937 |
| Test | 6.2722 | 7.8616 | 0.3031 |

Validation MAE is 2.94 minutes higher than training MAE, and validation RMSE
is 3.60 minutes higher. The R² drop from train to validation is 0.501. In
contrast, test MAE is only 0.020 minutes higher than validation MAE, and test
R² is 0.009 higher. This is a substantial train-to-held-out gap, but similar
validation and test results on adjacent future periods.

### Actual-versus-predicted and residual diagnostics

The existing held-out actual-versus-predicted chart,
[regression_actual_vs_predicted.png](./outputs/ml/plots/regression_actual_vs_predicted.png),
shows predictions compressed toward the middle relative to the diagonal:
shorter delivery times tend to be overpredicted and longer times tend to be
underpredicted.

Residuals below are `actual - predicted`; negative values mean overprediction.
On the test partition:

- Mean residual: **-0.323 minutes**; median: **-0.413 minutes**.
- Residual 5th and 95th percentiles: **-13.24** and **13.08 minutes**.
- Mean absolute error rises across predicted-time quartiles from **4.98** to
  **7.46 minutes**.
- Mean residual by predicted-time quartile declines from **+0.35** to
  **-1.07 minutes**, indicating some overprediction at the high end. The
  saved residual plot also shows broad, non-constant residual spread.

The saved residual analysis is available at
[regression_residual_analysis.png](./outputs/ml/plots/regression_residual_analysis.png).
The errors are centered near zero overall, but their magnitude and bias vary
with predicted time; average error alone therefore hides weaker performance
for some delivery-time ranges.

### Feature importance

The saved permutation importance is based on the held-out test sample, using
increase in MAE when a raw input is permuted. It is diagnostic, not causal:

| Input feature | Mean increase in MAE | Std. dev. |
|---|---:|---:|
| `Time_Orderd` | 1.1592 | 0.0677 |
| `distance_km` | 0.5931 | 0.0151 |
| `Festival` | 0.4153 | 0.0220 |
| `City` | 0.2709 | 0.0447 |
| `Type_of_order` | 0.0299 | 0.0072 |
| `Order_Date` | 0.0193 | 0.0136 |

Source: [regression_permutation_importance.csv](./outputs/ml/feature_importance/regression_permutation_importance.csv).
The small importance of order type and date means this fitted model gained
little measured test MAE from those inputs in this evaluation; it does not
prove that they can never be useful.

### Diagnosis

**Confirmed: the regression forest shows overfitting relative to its training
data, with moderate held-out predictive performance.** Training errors are
about half the validation errors and training R² is much higher, while
validation and test scores are close. It is **not showing the pattern expected
of a uniformly underfit model**: it fits the training partition substantially
better than the held-out partitions. The held-out R² near 0.30 nevertheless
means much of the target variation remains unexplained, so the model has
limited predictive strength outside the training period.

## 2. Classification: RandomForestClassifier

The positive class is `is_late = 1`, defined as delivery time greater than the
project's **30-minute analytical threshold**. It is not an official customer
promise.

### Threshold-independent performance

| Partition | Late prevalence | ROC-AUC | PR-AUC (average precision) |
|---|---:|---:|---:|
| Train | 30.10% | 0.9977 | 0.9945 |
| Validation | 28.07% | 0.7786 | 0.5817 |
| Test | 30.58% | 0.7777 | 0.5995 |

For context, a no-skill PR-AUC reference is approximately the positive-class
prevalence of each partition (30.10%, 28.07%, and 30.58%, respectively).
Validation and test ROC-AUC differ by less than 0.001; test PR-AUC is 0.018
higher than validation. Both held-out ranking metrics are materially below
the training metrics.

### Threshold-dependent performance

#### Saved threshold: 0.29

| Partition | Precision | Recall | F1 | Accuracy | Balanced accuracy | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Train | 0.6993 | 1.0000 | 0.8231 | 0.8706 | 0.9074 | `[[17508, 3978], [0, 9253]]` |
| Validation | 0.4466 | 0.7898 | 0.5705 | 0.6663 | 0.7040 | `[[3346, 2067], [444, 1668]]` |
| Test | 0.4607 | 0.8309 | 0.5927 | 0.6508 | 0.7012 | `[[2908, 2180], [379, 1862]]` |

The existing threshold was selected on validation data. On test data, 0.29
gives higher recall and lower precision than 0.50; this is a recall-oriented
trade-off with more false alarms. Test metrics are reported only as an
evaluation of the already selected threshold.

#### Default threshold: 0.50

| Partition | Precision | Recall | F1 | Accuracy | Balanced accuracy | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Train | 0.9479 | 0.9778 | 0.9627 | 0.9772 | 0.9774 | `[[20989, 497], [205, 9048]]` |
| Validation | 0.5457 | 0.5062 | 0.5252 | 0.7431 | 0.6709 | `[[4523, 890], [1043, 1069]]` |
| Test | 0.5644 | 0.5337 | 0.5486 | 0.7315 | 0.6761 | `[[4165, 923], [1045, 1196]]` |

The near-perfect training threshold metrics contrast with materially lower
validation and test metrics. Accuracy alone is not used to diagnose this
model; the large train-to-validation gaps in both ROC-AUC and PR-AUC, along
with the threshold-dependent metrics, provide the relevant evidence.

### Diagnosis

**Confirmed: the classification forest is strongly overfit to its training
partition.** Training ROC-AUC is 0.9977 and PR-AUC is 0.9945, versus validation
ROC-AUC 0.7786 and PR-AUC 0.5817. The held-out validation and test results are
close, so this is not evidence that the final seven-day test period suddenly
collapsed relative to validation; rather, the very high training scores do
not carry over to future dates.

The result is **not classified as underfitting**: the training performance is
near-perfect, not uniformly poor. Test accuracy is not the basis for this
conclusion.

## 3. Possible causes and leakage review

### Confirmed observations

- **Model capacity:** both forests have 140 trees, `min_samples_leaf=2`,
  `max_features=0.8`, and no explicit maximum depth (`max_depth=None`). That
  is a high-capacity configuration relative to a small set of raw inputs; the
  classification training metrics are consistent with the forest fitting
  training-specific structure.
- **Feature set:** both tasks use the same six order-placement candidates:
  `Order_Date`, `Time_Orderd`, `Type_of_order`, `City`, `Festival`, and
  `distance_km`. The pipeline creates calendar and cyclical order-time
  features, missingness indicators for date/time, imputes numeric values, and
  one-hot encodes categories. The forest receives 23 transformed columns.
- **Missingness:** distance is missing in **9.94%** of train rows and **6.83%**
  of validation and test rows. Order time is missing in **3.85%**, **3.72%**,
  and **3.64%**, respectively. The pipeline imputes distance with the
  training median; missing order time is represented by engineered missing
  values and an indicator before fitted preprocessing.
- **Target variability:** delivery-time standard deviation is about **9.3 to
  9.4 minutes** in each split. The 10th-to-90th percentile range is
  **15–40 minutes** across the partitions, wider than the typical prediction
  error. Late-order prevalence is **30.10%** in train, **28.07%** in
  validation, and **30.58%** in test.
- **Class imbalance:** approximately 70% of orders are on-time and 30% are
  late. The classifier uses `class_weight="balanced_subsample"`, but class
  weighting does not prevent overfitting.
- **Chronological distribution changes:** target mean and prevalence are
  broadly similar across the partitions. However, median distance changes
  from **9.19 km** in train to **7.77 km** in validation and **10.59 km** in
  test. This is evidence of some covariate shift; it is not, by itself, proof
  that distance shift caused the performance gap. City, festival, and order
  type proportions are comparatively similar between the splits.
- **Leakage checks:** the training notebook excludes the targets, order ID,
  `Time_Order_picked`, weather, traffic, rider, and vehicle fields from model
  inputs. It asserts disjoint order IDs and dates across chronological
  partitions. The saved pipelines transform the six approved input fields,
  and the notebook records that preprocessing is fitted on train only.
- **Input availability caveat:** the Module 02 feature notes treat
  `distance_km` as a placement-time candidate only if the selected
  restaurant/store and customer coordinates are known before prediction.
  Operational timing of store selection should be verified before deployment.
  This is an availability assumption, not evidence that the current saved
  model has target leakage.
- **Weather and traffic:** these fields were excluded because their
  availability at order-placement time was not verified. Their exclusion
  limits information available to the model but is appropriate for leakage
  control.

### Hypotheses, not proven causes

- The train-to-held-out gaps are consistent with an overly flexible forest
  learning training-specific patterns, particularly for classification.
  Establishing the contribution of model complexity would require controlled
  future experiments.
- Distance-distribution changes, missing input values, omitted operational
  predictors (such as verified traffic/weather at order placement, actual
  store/route conditions, or fulfillment workload), and differences between
  restaurant delivery and quick-commerce may contribute to held-out error.
  The current diagnostics do not identify their causal contribution.
- Regression's compressed predictions and growing absolute errors at higher
  predicted times are consistent with regression-to-the-mean behavior and
  possibly greater variability among longer deliveries. They do not prove a
  specific causal mechanism.
- The project uses food-delivery data as a proxy for quick-commerce
  operations; performance on actual quick-commerce orders remains unknown.

### Leakage conclusion

**No direct target or post-order leakage was identified in the declared
features or saved pipeline.** The binary target is derived from delivery time,
but neither `delivery_time` nor `is_late` is an input. The supplied checks
cannot independently prove every upstream field's operational timestamp;
the most important availability condition to verify is whether distance can
be computed before the prediction is made.

## 4. Future experiment recommendations

These are recommendations only; no experiment was run for this diagnostic.

1. Compare constrained forests (for example, shallower `max_depth`, larger
   `min_samples_leaf`, and varied `max_features`) with regularized or
   boosting alternatives. Use rolling-origin or forward-chaining validation;
   never use the final test period to tune.
2. Expand inputs only when their provenance and prediction-time availability
   are confirmed. Candidate operational data could include store workload,
   route/road conditions, or weather snapshots captured before prediction.
   Preserve a strict post-order exclusion list.
3. Measure feature drift and performance by date, distance range, and
   important categories in subsequent validation windows. Track missingness
   explicitly and test whether a distance-missing indicator helps rather
   than assuming median imputation is sufficient.
4. For classification, retain ROC-AUC and PR-AUC as threshold-independent
   measures. Define threshold selection against an explicit operational
   objective or business costs using validation data only; the current 0.29
   threshold is not claimed to be cost-optimal. Calibrate risk scores only
   with an appropriate validation procedure, then evaluate once on a locked
   future test set.
5. Validate both models on actual quick-commerce records before operational
   deployment. Report calibration and error slices alongside aggregate
   metrics.

## 5. Diagnostics not performed

A learning curve was not generated. Producing one would require fitting
repeated fresh models at multiple training-set sizes, which would violate the
diagnostic-only instruction not to retrain. Existing saved predictions,
plots, feature-importance tables, and metrics were inspected without
regenerating them.
