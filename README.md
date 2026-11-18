# Restaurant-Star-Rating-for-Zomato-Restaurant

#  Zomato Restaurants: Predicting Aggregate Rating

An end-to-end, **leakage-safe** machine-learning project that predicts a restaurant's Zomato aggregate rating from its listing attributes, flags "highly rated" restaurants, and scores restaurants that have no rating yet (the cold-start problem).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Status](https://img.shields.io/badge/Notebook-End--to--End-brightgreen)

---

##  Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Key Insights](#-key-insights)
- [Leakage & Overfitting Safeguards](#-leakage--overfitting-safeguards)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Next Steps](#-next-steps)
- [Author](#-author)

---

##  Problem Statement

On a restaurant-discovery platform, the aggregate rating is the most visible quality signal, yet:

1. **Cold start:** 2,148 of 9,551 listings (~22%) are *Not rated*, so they cannot be ranked or recommended well.
2. **Unclear quality drivers:** Which listing attributes (price tier, cuisine, location, table booking, online delivery) go together with high ratings?

**ML formulation**

| Task | Definition |
|---|---|
| **Primary (regression)** | Predict `Aggregate rating` (1.8 to 4.9) from listing attributes |
| **Secondary (classification)** | Is a restaurant *highly rated* (rating ≥ 4.0)? |
| **Scenario B: Cold-start** | Only attributes known at listing time (no votes) |
| **Scenario A: Established listing** | Also uses `Votes` (crowd engagement) |
| **Metrics** | RMSE (primary), MAE, R²; ROC-AUC / PR-AUC / F1 for classification |
| **Baseline** | Predict the training mean (RMSE ≈ 0.556) |

> Rating `0` means *Not rated* (a missing target), not a bad restaurant. These rows are excluded from training and scored by the model afterwards.

---

##  Dataset

- **File:** `Zomato_Restaurant_Dataset.csv`
- **Size:** 9,551 restaurants · 15 countries · 21 columns (7,403 rated after cleaning)
- **Coverage:** ~91% of rows are from India

**Data quality issues found and fixed**

| Issue | Fix |
|---|---|
| `Rating color` / `Rating text` are deterministic labels of the target | Dropped (leakage) |
| Costs in 12 currencies | Currency-free price features (cost relative to country median) |
| Wrong currency label for the Philippines ("Botswana Pula") | Remapped from country code |
| 499 restaurants with coordinates `(0, 0)`; geocoding outliers > 300 km from city | Set to `NaN`, imputed inside the pipeline with a missing flag |
| 18 restaurants with cost = 0 | Set to `NaN`, imputed in-pipeline |
| Corrupted characters in city names and the £ label | Regex repair |
| Duplicate listings, IDs, free-text addresses, scrape-time snapshot flags | Dropped |

---

##  Approach

```
raw CSV → audit & clean → split FIRST (80/20, stratified by rating band)
        → scikit-learn Pipeline (all learned statistics fitted on training folds only)
              feature builder → target-encode / impute / scale → feature selection → model
        → 5-fold CV model comparison → tuning (randomised search) → convex blend
        → fit diagnostics → leakage audit → single look at the test set
```

**Notebook sections**

1. Problem statement · 2. Proposed solution · 3. Data loading · 4. Data-quality audit & cleaning · 5. EDA & statistical tests · 6. Train/test split · 7. Feature engineering · 8. Feature selection · 9. Scaling · 10. Model comparison · 11. Tuning & ensembling · 12. Under/over-fitting diagnostics · 13. Leakage audit · 14. Final test evaluation · 15. Classification extension · 16. Scoring un-rated restaurants · 17. Recommendations

**Engineered features (~48, reduced to 40 via Lasso)**

- **Currency-free price level:** `ln(cost / country median)` and a within-country z-score
- **Geography:** haversine distance to the city centre, with centroid imputation and a missing-coordinates flag
- **Market structure:** leave-one-out city, locality and chain counts (avoids train/serve skew)
- **Cuisine information content:** mean and max surprisal (`-log₂ p(c)`), cuisine count, multi-hot top-25 cuisines
- **Cross-fitted Bayesian-shrinkage target encoding** for country, city and locality
- **Interactions:** price × India, booking × price

**Models compared:** Ridge, Lasso, Elastic Net, KNN, SVR, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, MLP, followed by tuned boosting and a convex blend.

---

##  Key Results

### Regression (held-out test set)

| Scenario | Model | RMSE (95% CI) | MAE | R² | vs. baseline |
|---|---|---|---|---|---|
| **B: Cold-start** (no votes) | Convex blend (HistGB / SVR / RF / Ridge) | **0.372** [0.356, 0.389] | 0.28 | 0.55 | ~33% better |
| **A: Established** (with votes) | Tuned HistGradientBoosting | **0.333** [0.315, 0.352] | 0.24 | 0.64 | ~40% better |

Baseline (predict the mean): RMSE 0.556. Ridge reference: 0.395 (B) / 0.352 (A).

### Classification: "highly rated" (≥ 4.0, ~19% of restaurants)

| Scenario | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| B: Cold-start | **0.90** | 0.70 | 0.66 |
| A: With votes | **0.93** | 0.78 | 0.71 |

The decision threshold was chosen on out-of-fold training predictions, never on the test set.

---

##  Key Insights

- **Location dominates in the cold-start model.** Shuffling `city` costs +0.127 RMSE, well above `cuisines` (+0.044) and `locality` (+0.035).
- **Votes are the single strongest signal** when available (+0.166 RMSE when shuffled; Spearman ρ ≈ +0.68 with rating).
- **Price tier and table booking** are positively associated with rating, both inside and outside India.
- **Online delivery shows no positive association** among rated restaurants. The raw gap on the full data came from un-rated restaurants rarely offering delivery.
- **India's average rating (3.35)** is the lowest in the sample; other countries sit at 3.6 to 4.5, probably reflecting a non-random "best-of" sample outside India.
- These are **associations, not causal effects**. Validate with an A/B or staggered roll-out before making business claims.

---

##  Leakage & Overfitting Safeguards

- **Split before learning:** the 20% test set stays sealed until Section 14.
- **Adversarial validation** (AUC 0.49) and a KS test confirm train and test are indistinguishable.
- **Everything learned lives inside the Pipeline** and is re-fitted per CV fold.
- **Six leakage audits:** target look-alike removal, naive-encoding counter-example (inflated CV R² of 0.91 vs. an honest 0.51), shuffled-target test (R² ≈ −0.05), chain-grouped CV, near-duplicate check, and a feature-timing checklist.
- **Fit diagnostics:** learning curves and boosting-stage curves show no under- or over-fitting. Test error is only +2% to +3% above CV error.

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```
> A recent scikit-learn is required (the notebook uses `TargetEncoder` and `root_mean_squared_error`; version 1.4+).

### 3. Add the data
Place the dataset next to the notebook.

>  The notebook reads `Zomato Restaurant Dataset.csv` (with spaces). Rename your file to match, or update the `pd.read_csv(...)` call in the data-loading cell.

### 4. Run the notebook
```bash
jupyter notebook Zomato_Restaurant_Rating_ML.ipynb
```
Choose **Kernel → Restart & Run All**. For a fast smoke test, set the environment variable `NB_QUICK=1` before launching.

### Output
`unrated_restaurant_predicted_ratings.csv` contains predicted ratings for the 2,148 un-rated restaurants.

---

##  Repository Structure

```
├── Zomato_Restaurant_Rating_ML.ipynb        # Full end-to-end notebook
├── Zomato_Restaurant_Dataset.csv            # Raw dataset (add locally)
├── unrated_restaurant_predicted_ratings.csv # Generated predictions
└── README.md
```

---

##  Limitations

- ~91% of rows are from India, so results for other countries rest on small, probably non-random samples.
- Random-split test error is slightly optimistic for brand-new chains (chain-grouped CV: +3% RMSE for B, +6% for A).
- The model compresses extreme ratings toward the mean (~3.4).
- There is no time dimension, so temporal drift cannot be assessed.
- Predictions for un-rated restaurants are expectations, not truths. Within-country ranking is more trustworthy than cross-country ranking.

**Recommended use:** triage and tiering of new listings (e.g. "likely ≥ 4.0"), not a customer-facing rating.

---

##  Next Steps

- Quantile / conformal prediction intervals
- Hierarchical (mixed-effects) models with city and chain random effects
- Spatial neighbour features
- NLP on review text (sentiment and topics)
- SHAP explanations for account managers
- Classifier probability calibration

---

## 👤 Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

Feel free to star the repo if you found it useful, and open an issue for questions or suggestions.
