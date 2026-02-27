# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

## Overview
This project analyses how market sentiment (Fear & Greed Index) influences crypto trader performance and behaviour. The analysis covers data preparation, statistical testing, trader segmentation, actionable strategy rules, and bonus predictive modelling.

## Executive Summary

### Methodology
We merged 211,226 trade-level records (32 accounts, Dec 2024 – Apr 2025) with daily Fear & Greed Index scores via a same-day left join on date. After timestamp normalization and cleaning (99.997% match rate), we engineered per-account daily metrics — PnL, win rate, trade frequency, position size, long/short ratio, and rolling 7-day drawdown. Trades were bucketed into **Fear** (FGI < 46) and **Greed** (FGI ≥ 55) regimes. We segmented traders along three dimensions: position size (Low/Med/High at 33rd/66th percentile), trade frequency (Infrequent/Moderate/Frequent), and PnL consistency (>60% positive days = Consistent Winner). All comparisons use Mann-Whitney U tests paired with Cohen's d effect sizes; win/loss proportions use Chi-square tests. Bonus tasks include a Logistic Regression vs XGBoost binary classifier for next-day PnL direction and K-Means clustering on behavioral features.

### Key Insights
1. **Sentiment materially impacts PnL.** Median daily PnL on Greed days ($265) is 2.2× that of Fear days ($123). The difference is statistically significant (Mann-Whitney U test), confirming that sentiment is a measurable risk factor — not just noise.
2. **High-size traders suffer disproportionately on Fear days.** They experience amplified drawdowns with no compensating win-rate gain, suggesting that large positions during panic markets compound losses rather than creating opportunity.
3. **Consistent Winners behave counter-cyclically.** Their long/short ratio adjustments across sentiment regimes differ meaningfully from Consistent Losers, indicating that winning behaviour on Fear days can be identified and replicated.

### Strategy Recommendations

**Rule 1 — Reduce position size on Fear days.** High-size traders (top 33%) show significantly higher drawdowns on Fear days vs Greed days with no win-rate benefit. During Fear regimes (FGI < 46), reduce position sizing to Medium-tier levels to preserve capital.

**Rule 2 — Adjust directional bias by sentiment.** Consistent Winners shift their long/short ratio between Fear and Greed regimes differently from Consistent Losers. Novice/losing traders should study and replicate the winners' directional patterns rather than following herd momentum.

**Rule 3 — Reduce trade frequency during Fear.** Frequent traders (top 33% by daily count) show worse median PnL on Fear days compared to infrequent traders, suggesting that reactive/panic-driven overtrading erodes returns. Cut trade count on Fear days to avoid compounding losses.

---

## Datasets

| Dataset | File | Rows | Columns | Description |
|---|---|---|---|---|
| Historical Trader Data | `historical_data.csv` | 211,226 | 16 | Trade-level records (account, coin, price, size, side, PnL, fees, timestamps) from Dec 2024 – Apr 2025 |
| Fear & Greed Index | `fear_greed_index.csv` | 2,645 | 4 | Daily sentiment scores (0–100) with classifications (Extreme Fear → Extreme Greed) from Feb 2018 |

**Post-cleaning merged dataset:** 211,224 rows × 19 columns | 32 unique accounts | 480 unique trading days

## Key Findings

1. **Traders earn higher median daily PnL on Greed days ($265) vs Fear days ($123)** — a 2.2× gap, validated with Mann-Whitney U test and Cohen's d effect size.

2. **Position size amplifies sentiment-driven risk** — High-size traders experience disproportionate drawdowns on Fear days with no compensating win-rate improvement, suggesting sentiment-aware position sizing is critical for capital preservation.

3. **Consistent Winners adjust directional bias differently across sentiment regimes** — their long/short ratio shifts are distinct from Consistent Losers, providing a replicable pattern for improving trader behaviour.

## Project Structure

```
├── analysis.ipynb           # Main analysis notebook (all sections)
├── historical_data.csv      # Raw trader data
├── fear_greed_index.csv     # Raw sentiment data
├── README.md                # This file
├── build_notebook.py        # Notebook generator script
├── chart1_pnl_boxplot.png          # PnL distribution by sentiment
├── chart2_behavior_bars.png        # Trade frequency & size by sentiment
├── chart3_winrate_heatmap.png      # Win rate by segment × sentiment
├── chart4_scatter_pnl_vs_size.png  # PnL vs size scatter
├── chart_behavioral_boxplots.png   # Behavioral metrics box plots
├── chart_cluster_selection.png     # K-Means elbow/silhouette
├── chart_cluster_heatmap.png       # Cluster × sentiment heatmap
└── chart_feature_importance.png    # XGBoost feature importances
```

## Notebook Sections

| # | Section | Task |
|---|---|---|
| 1 | Data Audit | Quality report, missing values, distributions |
| 2 | Timestamp Alignment & Merge | Date normalization, left join, sentiment buckets |
| 3 | Feature Engineering | Daily PnL, win rate, drawdown, trader segments |
| 4 | Sentiment-Split Performance | Mann-Whitney U, Cohen's d, win rate comparison |
| 5 | Behavioral Change Analysis | Position size, frequency, long ratio by sentiment |
| 6 | Trader Segmentation | Size/Frequency/Consistency × sentiment cross-tabs |
| 7 | Visualizations | 4+ charts with clear questions answered |
| 8 | Strategy Recommendations | 3 data-backed, falsifiable trading rules |
| 9 | Bonus: Predictive Model | Logistic Regression & XGBoost (AUC-ROC reported) |
| 10 | Bonus: Clustering | K-Means with elbow/silhouette, cluster profiling |

## How to Reproduce

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost jupyter

# 2. Place both CSV files in the project directory

# 3. Run the notebook
jupyter notebook analysis.ipynb

# Or execute non-interactively:
jupyter nbconvert --execute analysis.ipynb --to html
```

## Dependencies
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, xgboost, jupyter
