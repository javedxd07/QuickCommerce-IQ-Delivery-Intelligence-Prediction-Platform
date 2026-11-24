# UI Redesign Notes

## Design and layout

- Replaced the generic page layout with the **DELIVERY INTELLIGENCE** analytics dashboard.
- Added a dark, grouped sidebar, active-page styling, package mark, and hackathon footer.
- Applied the requested light palette, Inter-first system font stack, consistent page headers, spacing, borders, and responsive KPI grids.
- Added reusable presentation components in `src/dashboard_components.py` and shared styles in `assets/styles.css`. The brand mark is in `assets/brand.svg`; `.streamlit/config.toml` sets the light theme.
- Redesigned all eight pages: Executive Overview, Delivery Analytics, Geographic Intelligence, Rider Intelligence, Delivery Time Prediction, Late Delivery Risk, Management Recommendations, and About the Project.

## Preserved project behavior

- The dashboard reads saved project artifacts and does not train or retrain models.
- Model loading uses `st.cache_resource`; static datasets and reports use `st.cache_data`.
- Prediction inputs remain the six approved order-placement features: `Order_Date`, `Time_Orderd`, `Type_of_order`, `City`, `Festival`, and `distance_km`.
- The saved classification threshold remains **0.29**. The late-risk result is labelled an estimated risk score, not a calibrated probability, and explains the recall/false-alarm trade-off.
- Geographic and rider views use saved project outputs. The map describes coordinate-defined demand concentrations—not confirmed dark-store locations—and retains the food-delivery proxy limitation.
- Exactly three recommendations are read from the saved PS8 recommendations; their evidence and projected benefits are presented separately.
- Existing prediction-service callers remain compatible. The dashboard can pass cached models to the prediction functions rather than loading them for each request.

## Validation performed

- Confirmed the live Streamlit app returned HTTP 200 for its health endpoint and home page.
- Navigated all eight pages; each rendered its expected heading with no Streamlit exception.
- Exercised both prediction forms with the sample order. The regression page returned an estimate of 28.6 minutes; the late-risk page returned a score of 0.624 and applied the 0.29 threshold.
- Confirmed the geographic Plotly map rendered, the minimum-orders filter updated successfully, and the supporting data tables remained available.
- Confirmed the recommendations page rendered three cards with 12 compact evidence items and no raw JSON display.
- Checked responsive layouts at 1440, 1280, 768, and 390 pixels. The KPI grid adapts from six desktop columns to three or two columns at narrower widths, and the page had no horizontal body overflow at the checked sizes. The mobile sidebar can be expanded for navigation.
- Verified all 26 paths listed in the PS6–PS8 output manifest exist.

## Known limitations

- The project uses restaurant food-delivery data as a proxy for quick-commerce operations; validate any operational decision with relevant quick-commerce data.
- The 30-minute delay lens is analytical, not an official delivery promise.
- The 0.29 late-risk threshold is recall-oriented. The score is not a calibrated probability, and the trade-off is not claimed to be cost-optimal.
- Geographic clusters are not confirmed store locations; locality names are unavailable, and DBSCAN density chaining can produce broad regions.
- Rider segments are exploratory and have weak separation. Rider tenure is not recorded, and segment membership should not be used for individual personnel decisions.
- Responsive validation was performed in a browser at the listed viewport widths; it does not represent a full cross-device or assistive-technology audit.
