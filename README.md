# Telecom Customer Churn Prediction

End-to-end machine learning pipeline for predicting customer churn in a telecom dataset of 100,000 customers and 100 features. The project covers the full data science lifecycle: from raw data exploration through ensemble modelling, business dashboard, and financial impact quantification.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Telecom_customer_churn.csv` | Raw dataset — 100,000 customers, 100 features |
| `Description.json` | Column descriptions and data dictionary |
| `notebook.ipynb` | Full pipeline notebook (20 sections, 97 cells) |
| `project_overview.html` | Interactive HTML report — executive summary, dashboards, ROI simulator |

---

## Project Overview

### Problem Statement

Customer churn costs telecoms 5-7x more per acquisition than retention. A 5% improvement in retention can lift profitability by 25-95%. This pipeline predicts, 30-60 days ahead, which customers are at high risk of churning — enabling proactive retention campaigns before the customer decides to leave.

**Primary metric: Recall** — a missed churner costs ~$200 in lost lifetime value. A false alarm costs ~$10 in an unnecessary retention offer. The 20:1 cost asymmetry means catching churners matters far more than avoiding false alarms.

**Secondary metrics:** F1 Score, AUC-ROC, Precision

---

## Pipeline — 20 Sections

### Phase 1 — Data Preparation (Sections 1-11)

**Section 1 — Business Understanding**
Defines the problem, cost asymmetry, and metric hierarchy. Recall is the north star.

**Section 2 — Import Libraries**
Full dependency stack: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, imbalanced-learn, xgboost, lightgbm. Global `SEED = 42` for full reproducibility.

**Section 3 — Load Dataset**
100,000 rows x 100 columns. Column descriptions loaded from `Description.json`.

**Section 4 — Data Understanding**
- Data types audit (numerical vs categorical counts)
- Missing value rates per column
- Cardinality check
- Target class distribution: ~85% not churned / ~15% churned — confirmed imbalanced

**Section 5 — Exploratory Data Analysis**

*5.1 Univariate Analysis*
- Missing value bar chart with colour-coded thresholds (red >40%, orange 20-40%, blue <20%)
- Box plots for 10 key numerical features: `rev_Mean`, `mou_Mean`, `totmrc_Mean`, `ovrmou_Mean`, `custcare_Mean`, `drop_vce_Mean`, `months`, `hnd_price`, `eqpdays`, `avgrev` — reveals heavy right skew on revenue and usage columns, motivating Winsorization
- Bar charts for 11 categorical features showing frequency distributions

*5.2 Bivariate Analysis vs. Churn*
- Side-by-side box plots for each numerical feature split by churn status — `eqpdays` and `rev_Mean` show the clearest separation
- Churn rate bar charts per categorical feature with overall average reference line — high churn rates flagged in red (>15%)

*5.3 Statistical Significance Tests*
- **Welch's t-test** on all numerical features: 64/77 significant at p < 0.05. Top signals: `eqpdays` (t = -35.85), `hnd_price` (t = 32.67). Churners have older, cheaper phones.
- **Chi-Square test** on all categorical features: 15/21 significant. `asl_flag` and `crclscod` dominate.
- Key finding: every revenue and voice usage metric is lower in churners — early disengagement is observable before churn

*5.4 Correlation Analysis*
- Pearson heatmap on the 20 features most correlated with churn
- Highly correlated pairs (|r| > 0.90) flagged — revenue columns are heavily redundant with each other, motivating the correlation filter in Section 8

**Section 6 — Data Cleaning**
Decision matrix applied systematically:
- Columns >40% missing: dropped
- Categorical 20-40% missing: filled with "Unknown" label
- Numerical missing: median for skewed distributions (|skew| > 1), mean for normal
- Winsorization at 1st-99th percentile on all revenue and usage columns — extreme telecom users are real outliers, not data errors

**Section 7 — Feature Engineering**
8 domain-driven features created from raw columns:

| Feature | Formula | Business Signal |
|---------|---------|-----------------|
| `call_completion_rate` | `comp_vce_Mean / (plcd_vce_Mean + 1)` | Failed calls = customer frustration |
| `overage_ratio` | `ovrmou_Mean / (mou_Mean + 1)` | Being charged extra = churn risk |
| `rev_per_minute` | `rev_Mean / (mou_Mean + 1)` | Revenue efficiency signal |
| `high_custcare` | 1 if `custcare_Mean` > Q75 | Frequent complaints = dissatisfied |
| `drop_block_rate` | `drop_blk_Mean / (attempt_Mean + 1)` | Network quality signal |
| `rev_trending_down` | 1 if `change_rev` < 0 | Revenue decline before churn |
| `tenure_group` | Cut into 0-6m, 6-12m, 12-24m, 24-48m, 48m+ | Non-linear tenure effect |
| `eqp_age_group` | Cut into <6m, 6m-1y, 1-2y, 2y+ | Device age risk signal |

Encoding strategy: ordinal features via LabelEncoder, high-cardinality nominal via frequency encoding, remaining nominal via one-hot encoding (drop_first=True).

**Section 8 — Feature Selection**
Four-stage funnel reducing 100+ features to 25:

1. **VarianceThreshold (0.01)** — removes near-constant features
2. **Correlation filter (|r| > 0.90)** — removes redundant pairs, keeping the more informative one
3. **Mutual Information top 40** — model-agnostic, captures non-linear relationships that Pearson misses
4. **Random Forest importance top 25** — final selection based on tree-based impurity gain

Output: MI score bar chart (top 10 highlighted in red) and RF importance horizontal bar chart.

**Section 9 — Train / Test Split**
80/20 stratified split. Stratification ensures churn rate in train and test sets match within 0.5%.

**Section 10 — Handle Class Imbalance (SMOTE)**
If training churn rate < 30%: SMOTE synthesises new minority-class samples from k-nearest neighbours on training data only — no synthetic samples ever enter the test set. Otherwise: `class_weight='balanced'` in each model.

**Section 11 — Scaling**
Two data versions maintained:
- `X_train_scaled / X_test_scaled` — StandardScaler for LR and KNN
- `X_train_tree / X_test_tree` — unscaled for all tree-based models

Scaler fitted on training data only, then applied to test — no leakage.

---

### Phase 2 — Baseline Modelling (Sections 12-14)

**Section 12 — Logistic Regression (Baseline)**
- Solver: lbfgs, C=1.0, class_weight='balanced', max_iter=1000
- Interpretable via log-odds coefficients
- Output: confusion matrix + PCA 2D decision boundary scatter plot + 6-metric interpretation panel
- Role: sets the performance floor for all subsequent models

**Section 13 — Multiple Baseline Models**

Each model produces: confusion matrix, PCA 2D class distribution with decision boundary, and a metric panel with colour-coded badges (green/orange/red by threshold).

| Model | Key Parameters | Data Version |
|-------|---------------|--------------|
| Decision Tree | max_depth=5, class_weight='balanced' | Unscaled |
| KNN | k=11, n_jobs=-1 | Scaled |
| Random Forest | 200 trees, max_depth=12, class_weight='balanced' | Unscaled |
| XGBoost | 200 estimators, learning_rate=0.05, scale_pos_weight=imbalance ratio | Unscaled |

**Section 14 — Hyperparameter Tuning**
RandomizedSearchCV with 25 iterations x 5-fold StratifiedKFold, scoring on F1:

*Random Forest search space:* n_estimators [100-400], max_depth [6-None], min_samples_split, min_samples_leaf, max_features, class_weight

*XGBoost search space:* n_estimators, max_depth [3-8], learning_rate [0.01-0.15], subsample, colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda

All tuned models added to the shared `results` dict for unified comparison in Section 16.

---

### Phase 3 — Advanced Ensembles (Section 15)

**Section 15.1 — LightGBM (Tuned)**

LightGBM uses **leaf-wise tree growth** instead of the level-wise approach used by XGBoost. On wide datasets (100 features), this gives 10-20x faster training and often finds better splits because it targets the leaf with the highest loss reduction — not the next level uniformly.

Key parameter: `num_leaves` controls model complexity independently of depth. A tree with `max_depth=6` and `num_leaves=63` is far more expressive than a conventional depth-6 tree.

Search: 50 iterations x 5-fold CV over n_estimators [300-1000], learning_rate [0.01-0.1], num_leaves [31-127], max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda.

Target: F1 > 0.65, AUC > 0.70, Recall > 0.70

**Section 15.2 — Soft Voting Ensemble (RF + XGBoost + LightGBM)**

Soft voting averages **predicted probabilities** across three models — not class labels. This preserves each model's confidence signal. The three base models make structurally different errors:
- Random Forest: bagging reduces variance, struggles with fine decision boundaries
- XGBoost: level-wise boosting, strong on structured patterns
- LightGBM: leaf-wise boosting, finds finer non-linear splits

`weights=[1, 2, 2]` gives the two boosters double influence over RF, reflecting their individual superiority on this dataset.

Target: F1 > 0.66, AUC > 0.71, Recall > 0.71

Output includes a side-by-side comparison vs. the best single model (XGBoost) showing per-metric deltas.

**Section 15.3 — Stacking Classifier**

Stacking is the most sophisticated ensemble technique used. It solves a fundamental problem with voting: **we don't know the right weights** — they may vary by customer segment. Stacking learns them from data.

*How it works:*

1. The training set is split into 5 folds using StratifiedKFold
2. For each fold: the four base models (Decision Tree, RF Tuned, XGBoost, LightGBM) are trained on the other 4 folds and used to predict on the held-out fold
3. This generates **out-of-fold (OOF) predictions** — predicted probabilities for every training sample, from a model that never saw that sample
4. The OOF predictions form a new feature matrix (N_train x 4) — one probability column per base model
5. A **Logistic Regression meta-learner** is trained on this OOF matrix to learn the optimal combination
6. At inference time: all four base models predict on the test set, and the meta-learner combines their probabilities

*Why OOF matters:* If base models predicted on data they were trained on, the meta-learner would see artificially confident predictions and overfit to the training distribution. OOF predictions are honest — they simulate test-time behaviour.

*Meta-learner output:* The LR coefficients per base model are printed and visualised as a horizontal bar chart. Positive coefficients = the meta-learner trusts that model's churn probability. Negative coefficients = the meta-learner inversely weights it (rare, indicates redundancy). This chart directly answers: which model does the stacker trust most per customer?

*Why Logistic Regression as the meta-learner:*
- Interpretable: coefficients are auditable
- Regularised: avoids overfitting the OOF predictions
- Fast: the meta-feature matrix is only N x 4
- `class_weight='balanced'` preserves recall priority

Target: F1 > 0.67, AUC > 0.72, Recall > 0.72

**Section 15.4 — Threshold Optimisation**

Every classifier outputs a probability between 0 and 1. The **decision threshold** is the cutoff above which a customer is flagged as a churner. The default is 0.5, but this ignores the cost asymmetry of the problem.

*The cost argument:*
- False Negative (missed churner): ~$200 loss
- False Positive (unnecessary offer): ~$10 cost
- Optimal threshold should reflect this 20:1 ratio — lower than 0.5

*Method:* Sweep thresholds from 0.10 to 0.90 in steps of 0.01. At each threshold, compute F1, Recall, and Precision. Select the threshold that maximises F1.

*Output — two charts:*

**Chart 1 — Threshold vs Metric curves:**
- X-axis: decision threshold (0.10 to 0.90)
- Y-axis: metric score (0 to 1)
- Three lines: F1 (blue), Recall (red), Precision (green)
- Vertical dashed orange line: optimal threshold (F1-maximising)
- Vertical dotted grey line: default threshold (0.50)
- The crossing of Recall and Precision lines shows the Precision-Recall tradeoff point
- As threshold decreases: Recall rises (catch more churners), Precision falls (more false alarms)
- As threshold increases: Precision rises, Recall falls

**Chart 2 — Precision-Recall curve:**
- X-axis: Recall (0 to 1)
- Y-axis: Precision (0 to 1)
- The curve traces all possible (Recall, Precision) pairs as the threshold varies
- Orange dot: optimal threshold operating point
- Grey diamond: default 0.50 threshold operating point
- Ideal model hugs the top-right corner (high Recall AND high Precision)
- Area under this curve (AUPRC) is often more informative than AUC-ROC on imbalanced data

*The threshold as a business dial:* Once trained, the model never needs retraining to adjust aggressiveness. Lower the threshold to catch more churners (higher campaign cost, higher revenue recovery). Raise it to save budget (miss more churners, lower spend). This decision belongs to the business, not the data scientist.

The final result is stored as `Best + Threshold` in the results dict: highest-AUC bonus model + optimal data-driven threshold.

Target: F1 > 0.68, AUC > 0.72, Recall > 0.78

---

### Phase 4 — Evaluation & Business Output (Sections 16-20)

**Section 16 — Full Model Evaluation & Comparison**

All 11 models compared in a single unified table sorted by F1 Score. The `results` dict accumulated outputs throughout the notebook — this section reads it directly, no duplication.

*Comparison table columns:* Model, Type (Baseline / Ensemble), Accuracy, Precision, Recall, Specificity, F1 Score, AUC-ROC, Train-Test Gap

*ROC curve chart:*
- X-axis: False Positive Rate (1 - Specificity), range 0 to 1
- Y-axis: True Positive Rate (Recall), range 0 to 1
- Each model is one curve; ensemble models drawn bold
- Best + Threshold in red, other ensembles in orange, baselines in muted blue tones
- Diagonal dashed line: random classifier (AUC = 0.50)
- A model dominates if its curve is above and to the left of another — higher AUC = better ranking ability at all thresholds
- The AUC score summarises the entire curve into one number: probability that a randomly chosen churner receives a higher score than a randomly chosen non-churner

*Comparison bar chart:*
- 5 subplots side by side: Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Each bar represents one model; all models on the same x-axis
- Red: best ensemble model, orange: other ensembles, blue: baselines
- Y-axis: 0 to 1.14 (extra headroom for value labels)
- Value labels printed above each bar; ensemble labels bold

**Section 17 — Feature Importance**

Gini importance from the best tree-based model by F1 score. Horizontal bar chart with 25 features, coloured on a red-blue gradient (most important = red).

Gini importance measures the total reduction in node impurity contributed by a feature across all trees and all splits. Higher = the model relied on that feature more. The 8 engineered features from Section 7 typically appear in the top 10.

**Section 18 — Business Dashboard (Interactive Plotly)**

Five interactive charts for non-technical stakeholders:

1. **Churn Probability Histogram** — distribution of model scores, coloured by actual churn. Two overlapping histograms (blue = not churned, red = churned). Well-separated distributions indicate good model discrimination.

2. **Risk Segment Pie** — customers binned into Low (0-30%), Medium (30-60%), High (60-100%) risk. Donut chart with percentage labels. The High Risk segment drives the retention campaign list.

3. **Interactive Feature Importance Bar** — top 15 features, horizontal bars coloured by importance score. Hover for exact values.

4. **Interactive ROC Curve** — all models overlaid, hover for exact FPR/TPR at each point. Unified tooltip shows all model values at the same FPR.

5. **Interactive Confusion Matrix Heatmap** — cells labelled with business context (True Negative = "Loyal - Correct", False Positive = "Loyal - Flagged", False Negative = "Churner - Missed", True Positive = "Churner - Caught") plus raw counts.

**Section 19 — Revenue at Risk Analysis**

Directly answers: how much revenue are we about to lose? The model's churn probabilities are joined back to the original revenue columns via the test set index.

*Revenue columns used:*
- `rev_Mean` — mean monthly revenue per customer (primary)
- `totrev` — total revenue over customer lifetime
- `adjrev` — billing-adjusted total revenue
- `avgrev` — average monthly revenue over lifetime
- `avg3rev`, `avg6rev` — recent 3 and 6 month revenue averages
- `ovrrev_Mean` — mean total monthly overage revenue
- `vceovr_Mean` — voice overage component
- `datovr_Mean` — data overage component
- `totmrc_Mean` — mean total monthly recurring charge

*19.1 — Revenue profile reconstruction:* Test set rows matched to `df_clean` by pandas index. Churn probability and predicted label attached. Fallback chain if `rev_Mean` missing: `avgrev` → `totmrc_Mean` → $50 placeholder.

*19.2 — Headline exposure numbers:*

**Method 1 — Hard revenue at risk:** Sum of `rev_Mean` for all predicted churners. Assumes every flagged customer would have churned with certainty. This is the upper bound.

**Method 2 — Probability-weighted expected loss:** For each customer: `churn_proba x rev_Mean`. Sum across all test customers. This is the actuarially correct expectation — a customer with 30% churn probability contributes 30% of their monthly revenue to the exposure figure.

**Method 3 — Annualised exposure:** Monthly at-risk figure multiplied by 12. Assumes the churner would have stayed for one more year if retained.

**Method 4 — Undetected leak (False Negatives):** Sum of `rev_Mean` for customers who actually churned but were not flagged by the model. This is invisible money leaving the business — no campaign will be sent to these customers. Quantifying it makes the case for improving Recall.

*19.3 — Revenue tier segmentation:* Predicted churners split into quartiles (Q1-Q4) by `rev_Mean`. For each tier: customer count, total monthly revenue at risk, average revenue, average churn probability, expected loss, percentage of total exposure, and `roi_priority` (expected loss per customer in that tier).

`roi_priority` is the key allocation metric: it tells the retention team how much expected monthly revenue is recovered per customer contacted in that tier, before retention offer cost. Spend budget proportionally to this score — not to customer count.

Three charts: customers per tier (bar), monthly revenue at risk per tier (bar), share of total exposure per tier (horizontal bar).

*19.4 — Early-warning cohort:* Customers where `avg3rev < avg6rev` — their revenue is already declining in the three months before the observation date. Intersection with predicted churners gives the highest-priority cohort: the model predicts churn AND an observable leading indicator confirms it. Two charts: histogram of revenue decline percentage, scatter of decline % vs churn probability.

*19.5 — Overage risk cohort:* Customers paying overage fees (`ovrrev_Mean > 0`) who are also predicted churners. Overage customers have higher ARPU and higher churn probability simultaneously — the most urgent intervention target. Business action: automatic plan-upgrade alert before overage is triggered (converts frustration source into a loyalty touchpoint, retains the elevated revenue). Two charts: revenue distribution overlay (overage vs non-overage churners), overage component breakdown by type (voice vs data).

*19.6 — Interactive Plotly dashboard (2x2):*
- Top-left: churn probability vs monthly revenue scatter, coloured by predicted class
- Top-right: expected monthly loss distribution histogram for predicted churners
- Bottom-left: customers per revenue tier bar chart
- Bottom-right: ROI priority per revenue tier bar chart

**Section 20 — Conclusion & Action Plan**

Full pipeline summary, model progression narrative, top churn signals with operational actions, revenue at risk action plan, and monitoring/maintenance specifications.

---

## Interactive HTML Report (`project_overview.html`)

Standalone self-contained HTML file. No server required — open in any browser.

**Executive Summary**
All KPIs in a single view. Business-framed confusion matrix with dollar context per cell (TP = revenue saved, FN = revenue lost, FP = offer cost, TN = no action needed). Full model leaderboard table. ROC curve and comparison bar charts pulled directly from notebook outputs.

**Data Analysis**
All EDA charts: class distribution, missing values, box plots, churn-by-category bar charts, correlation heatmap. Mirrors Section 5 of the notebook.

**Feature Engineering**
Mutual information top-40 bar chart and RF importance top-25 chart. Shows which raw and engineered signals carry the most predictive power before any model is trained.

**Baseline Models**
Confusion matrix outputs for all 6 baseline models with metric summary tables. Allows direct comparison of model behaviour at the classification level.

**Ensemble Models**
LightGBM, Voting Ensemble, and Stacking results. Includes the meta-learner coefficients bar chart — shows which base model the stacker trusts most for churn predictions.

**Threshold Optimisation**
The threshold curve chart from the notebook reproduced in HTML. Plus an interactive slider that lets the user move the threshold from 0.10 to 0.90 and watch F1, Recall, and Precision update live — making the business tradeoff immediately tangible.

**Revenue at Risk**
All four calculation methods explained in a side-by-side table (hard exposure, probability-weighted, annualised, undetected leak). The three-tier risk segmentation bar charts. Headline financial numbers in large format.

**ROI Simulator**
Four interactive sliders:
- Cost per retention offer ($)
- Retention success rate (%)
- Months of revenue saved per retained customer
- Target segment (all predicted churners / high risk only / early-warning cohort)

Outputs computed in real time: campaign cost, retained customers, revenue saved, net ROI ($), ROI ratio (x). Lets business and finance teams test scenarios before committing budget.

---

## Technical Details

### Data Integrity Controls

- `SEED = 42` enforced globally — all results fully reproducible
- SMOTE applied on training data only — no synthetic samples in test set
- StandardScaler fitted on training data only — no test set contamination
- Stacking uses StratifiedKFold OOF predictions — no meta-layer leakage
- Train-Test F1 gap monitored on all models — alert threshold 0.10

### Model Results Storage

All models write into a shared `results` dict with a consistent schema:

```python
{
    'model':        trained model object,
    'name':         model name string,
    'accuracy':     float,
    'precision':    float,
    'recall':       float,
    'specificity':  float,
    'f1':           float,
    'auc':          float,
    'train_f1':     float,
    'gap':          float,
    'y_pred':       np.ndarray,
    'y_proba':      np.ndarray or None
}
```

This allows Section 16 (comparison), Section 18 (dashboard), and Section 19 (revenue at risk) to all reference the same outputs without re-running any model.

### Metric Reference

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | (TP+TN) / N | Misleading on imbalanced data |
| Precision | TP / (TP+FP) | Of flagged churners, how many were real |
| **Recall** | **TP / (TP+FN)** | **Of all churners, how many were caught — PRIMARY** |
| Specificity | TN / (TN+FP) | Of loyal customers, correctly left alone |
| F1 Score | 2 x (P x R) / (P+R) | Harmonic mean — best single summary on imbalanced data |
| AUC-ROC | Area under ROC | Ranking ability across all thresholds; 0.5 = random, 1.0 = perfect |

### Production Monitoring

| KPI | Target | Action if breached |
|-----|--------|--------------------|
| Recall | >= 0.70 | Retrain on latest 3 months |
| AUC-ROC | >= 0.70 | Retrain — distribution shift detected |
| Threshold | Quarterly review | Re-run sweep if cost ratio changes |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
xgboost
lightgbm
scipy
```

Install all:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn xgboost lightgbm scipy
```

---

## Reproducing Results

1. Place `Telecom_customer_churn.csv` and `Description.json` in the same directory as `notebook.ipynb`
2. Install dependencies
3. Run all cells top to bottom — sections are ordered so every variable is available when needed
4. All charts render inline. The interactive Plotly charts in Section 18 and 19 require a live kernel (not nbviewer)
5. Open `project_overview.html` in any browser — no server required
