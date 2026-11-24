# Manthan Quick-Commerce Delivery Intelligence

## Project overview

This project analyzes delivery operations for the Manthan hackathon (KLU + Path-3). It covers data quality, delay definition and descriptive analysis, delivery-time and late-order modeling, geographic demand zones, rider-level segmentation, and management recommendations.

The available data represents **restaurant food delivery**, not quick-commerce grocery or instant-commerce operations. Results are findings about this dataset and should not be generalized to all quick-commerce businesses without validation on actual quick-commerce operational data.

## Business problem

The analysis asks how delivery duration and late-order risk vary, which order-placement inputs support prediction, where valid customer delivery locations form geographic demand concentrations, and how riders can be grouped by observed delivery outcomes. The analytical late threshold is 30 minutes. It is not an officially confirmed customer promise.

## Dataset

- Authoritative raw input: `data/train.csv`
- PS1 cleaned output: `data/cleaned_data.csv`
- PS1 geographic validation output: `data/final_cleaned_data.csv`
- PS2 model-ready output: `data/model_ready_data.csv`
- The redundant root-level `train.csv` was byte-identical to `data/train.csv` and was removed after confirming that all notebook source references use the authoritative `data/train.csv` path.
- The data contains delivery records, rider identifiers, customer and restaurant coordinates, order times, delivery time, ratings, and categorical fields. `City` contains service-category labels such as Metropolitan, Urban, and Semi-Urban; it is not a city or locality name. Rider tenure/years of experience are unavailable.

## Project architecture and notebook order

Run notebooks in order from the project root:

1. `notebooks/01_Data_Preparation.ipynb` — PS1: data quality checks and cleaned data.
2. `notebooks/02_Delay_Lens_and_EDA.ipynb` — PS2/PS3: 30-minute analytical delay lens, approved order-placement feature assessment, and descriptive analysis.
3. `notebooks/03_Machine_Learning.ipynb` — PS4/PS5: chronological regression and classification with train-fitted preprocessing, validation-based model selection, and reusable prediction pipelines.
4. `notebooks/04_Dark_Store_Rider_Intelligence_and_Recommendations.ipynb` — PS6/PS7/PS8: K-Means comparison, Haversine DBSCAN demand hotspots, exploratory rider segmentation, and exactly three recommendations.
5. `notebooks/05_Model_Optimization.ipynb` — validation-selected V2 model experiments and final held-out test evaluation. The dashboard loads its saved pipelines and does not retrain them.

Execute all notebooks from the project root so project-relative data and artifact paths resolve:

```powershell
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_Data_Preparation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/02_Delay_Lens_and_EDA.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_Machine_Learning.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_Dark_Store_Rider_Intelligence_and_Recommendations.ipynb --inplace
```

PS1 and PS2/PS3 outputs are prerequisites for later modules. Do not execute later notebooks before their prerequisite outputs are available. Model optimization is already complete; do not rerun it for the dashboard.

## Project structure

```text
app.py
data/                       # source and prepared datasets
models/                     # saved prediction and clustering artifacts
notebooks/                  # validated PS1–PS8 analysis notebooks
outputs/
  eda/ ml/                  # saved EDA, model metrics, charts, predictions
  ps2/ ps6/ ps7/ ps8/       # delay, clustering, rider, recommendation outputs
assets/
  brand.svg                 # dashboard wordmark
  styles.css                # responsive dashboard design system
src/
  dashboard_components.py  # reusable UI components
  dashboard_utils.py        # dashboard artifact paths and readers
  feature_engineering.py    # shared order-placement feature engineering
  prediction_service.py     # saved model loading and prediction functions
requirements.txt
README.md
```

## Problem statements

- **PS1 — Data Quality & Reliability:** validates missing values, duplicates, rider fields, GPS, timestamps, delivery time, and categories while preserving audit information.
- **PS2 — Define the Delay Lens:** applies the 30-minute analytical threshold (`delivery_time > 30` is late), not an official service-level promise.
- **PS3 — Diagnose Delay Patterns:** describes time, weather/traffic, distance, and rider/order patterns without causal claims.
- **PS4 — Delivery-Time Prediction:** compares a baseline, Ridge, and Random Forest using MAE, RMSE, and R².
- **PS5 — Late-Order Prediction:** compares baseline, Logistic Regression, and Random Forest; selects model and threshold with validation data, not test data.
- **PS6 — Geographic Demand Hotspots:** retains the broad K-Means comparison and uses DBSCAN with Haversine distance on valid customer coordinates. K-distance diagnostics guide the radius. DBSCAN noise is reported separately. Hotspots are demand concentrations, not confirmed optimal dark-store sites.
- **PS7 — Rider Segmentation:** aggregates rider ratings, delivery duration, late rate, and deliveries per observed active day; imputes missing rider-level mean ratings within the clustering pipeline. Segments are exploratory groups. Age is not treated as professional experience.
- **PS8 — Management Recommendations:** exports exactly three evidence-backed recommendations with projected benefits clearly separated from observed results.

## Machine-learning methodology

PS4 and PS5 use the six order-placement features from the Module 02 availability report: `Order_Date`, `Time_Orderd`, `Type_of_order`, `City`, `Festival`, and `distance_km`. Targets and post-order features are excluded. Chronological date partitions separate training, validation, and test data. Preprocessing is fitted on training data only, and the classification threshold is selected on validation data. The saved pipelines and prediction service are under `models/` and `src/`.

## Clustering methodology

### Geographic hotspots

PS6 filters to customer coordinates that passed PS1's delivery-coordinate validation; invalid or unavailable coordinates remain in the assignment file as unclustered records. The broad K-Means solution is retained as a comparison. DBSCAN uses customer latitude/longitude converted to radians and the Haversine metric. `eps` is selected from a k-distance curve; `min_samples` is assessed against duplicate-coordinate effects and cluster/noise diagnostics. Cluster summaries report order counts, demand share, mean delivery time, late rate, network differences, and geographic centroids.

In the current run, 5- and 10-neighbor distance curves were dominated by repeated zero-distance coordinates, while the 15-neighbor curve yielded a positive observed knee. The selected parameters were `min_samples=15` and `eps=1.561 km` (converted to radians for Haversine DBSCAN). The resulting 132 connected components contain 41,302 orders; 651 valid-GPS orders (1.55%) are reported as DBSCAN noise, and 3,640 invalid/unavailable customer GPS records remain unclustered. The valid-GPS network benchmark was 26.31 minutes average delivery time and 29.94% late.

DBSCAN `eps` bounds neighbor distance, not the full size of a connected component: the largest 95th-percentile radius from a component centroid is 11.39 km, with a maximum individual radius of 13.75 km. The chain-connectivity behavior and lack of actual city/locality identifiers limit fine-grained site planning. `City` values are service categories, so localities cannot be named or verified. Even a dense hotspot is not evidence that a new dark store is needed; local supply, inventory, road networks, and actual quick-commerce demand must be assessed separately.

### Rider segments

Rider groups are based on observed ID-level delivery statistics. The feature scaler and rating imputer are fitted with the K-Means pipeline. Rider age is not a proxy for rider tenure. A silhouette score around 0.25 indicates weak separation, so segment names are descriptive summaries, not discrete or permanent rider types.

## Outputs and models

- `outputs/ps2/`, `outputs/eda/` — delay and EDA reports.
- `outputs/ml/` — validation/test comparisons, metrics, charts, feature importance, and test predictions.
- `outputs/ps6/` — original K-Means assignments/profiles, DBSCAN hotspot assignments/summaries, parameter diagnostics, maps, and metadata.
- `outputs/ps7/` — rider assignments, segment profiles, cluster diagnostics, charts, and metadata.
- `outputs/ps8/` — exactly three recommendations in JSON and readable text formats.
- `outputs/ps6_ps7_ps8_output_manifest.json` — stable paths and applicability notes for future dashboard integration.
- `models/` — saved V1 backup regression/classification pipelines, V2 experimental pipelines under `models/experimental/`, and PS6/PS7 clustering estimators.
- `src/feature_engineering.py` and `src/prediction_service.py` — reusable PS4/PS5 transformations and inference functions.

## Streamlit dashboard

The Streamlit dashboard integrates the saved PS1–PS8 artifacts and does not execute notebooks or retrain models at application startup. Grouped navigation and a shared responsive design system provide the enterprise-style interface. It includes:

- Executive overview and validated project findings.
- Delivery analytics using saved EDA tables and charts.
- Delivery-time and late-risk predictions using the saved end-to-end pipelines.
- Geographic demand hotspots and rider segment summaries.
- The exactly-three PS8 management recommendations and project limitations.

Run the app from the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The prediction service uses the V2 experimental HistGradientBoosting pipelines under `models/experimental/`. Its classification threshold is read from `models/experimental/classification_threshold_v2.json` (0.48); the reported score is an estimated late-delivery risk score, not a calibrated probability. The original V1 model files remain untouched. Demo scenarios use complete observed rows from `data/model_ready_data.csv`. The six approved raw input features are passed to the saved pipelines, which contain their own feature engineering and preprocessing. The feature-availability report assumes distance can be computed before prediction from known selected-store and customer coordinates; verify this operational timing before deployment. Missing artifacts are shown as dashboard errors rather than replaced with synthetic values.

### Screenshots

Screenshots can be added here after capturing the application in the target deployment environment.

## Current key findings

The current Module 04 run selected a four-segment rider solution (silhouette **0.247**, weak separation). Its highest-late-rate exploratory segment contains 265 riders and has a 42.28% late rate, compared with a 29.99% average across rider-level rates. The three current recommendations are:

1. **High — Review the supported DBSCAN hotspot with the highest late rate:** 87 orders, 50.57% late versus a 29.94% valid-GPS network rate; use a local operational review, not an automatic footprint change.
2. **High — Offer non-punitive workflow support to the highest late-rate rider segment:** 265 riders, 7,592 deliveries, and a 42.28% segment-average late rate; the silhouette warns that segment membership is not definitive.
3. **Medium — Pilot a capacity review for the largest observed delivery hotspot:** 3,187 valid-GPS orders (7.60% share); validate locality-level current demand and supply before any deployment.

Benefits are projected, not observed. These measurements are descriptive and food-delivery-specific; inspect the JSON/text recommendation outputs for all supporting evidence and qualifications.

## Dataset limitations

- Food-delivery records are a proxy, not representative proof of quick-commerce grocery operations.
- `City` is a service-category label, not a geographical city name. No locality-level conclusions can be verified.
- Rider tenure/experience is not recorded.
- DBSCAN only finds density concentrations in the observed valid customer coordinates. The results depend on spatial radius and minimum density parameters.
- Zone centroids are not site-selection recommendations. Actual supply availability, travel networks, demand at finer time scales, and costs are not represented.
- Clustering and descriptive associations do not demonstrate causation.

The dashboard loads saved PS1–PS8 outputs and the V2 experimental prediction pipelines. It does not execute notebooks or fit models during startup.
