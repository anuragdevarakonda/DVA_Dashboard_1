# NovaMart Marketing Analytics Dataset

## 📊 Masters of AI in Business - Data Visualization Assignment

This dataset simulates 2 years (2023-2024) of marketing data for **NovaMart**, a fictional omnichannel retail company operating across India.

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `campaign_performance.csv` | 5,858 | Daily campaign metrics (impressions, clicks, conversions, spend, revenue) |
| `customer_data.csv` | 5,000 | Customer demographics, behavior, and churn indicators |
| `product_sales.csv` | 1,440 | Hierarchical product sales by category/subcategory |
| `lead_scoring_results.csv` | 2,000 | ML model predictions vs actual conversions |
| `feature_importance.csv` | 11 | Pre-calculated feature importance scores |
| `learning_curve.csv` | 11 | Training/validation scores at different data sizes |
| `geographic_data.csv` | 15 | State-level performance metrics with coordinates |
| `channel_attribution.csv` | 8 | Multi-touch attribution model comparison |
| `funnel_data.csv` | 6 | Marketing funnel stages and conversion rates |
| `customer_journey.csv` | 8 | Multi-touchpoint customer paths |
| `correlation_matrix.csv` | 10x10 | Pre-computed metric correlations |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Project Structure
```
your_project/
├── data/
│   ├── campaign_performance.csv
│   ├── customer_data.csv
│   ├── ... (all CSV files)
├── app.py (or streamlit_starter_app.py)
├── requirements.txt
└── README.md
```

### 3. Run the Dashboard
```bash
streamlit run streamlit_starter_app.py
```

---

## 📈 Data Insights Built Into Dataset

### Campaign Performance
- **Seasonality**: Diwali (Oct-Nov) and Christmas (Dec) show 30-40% revenue boost
- **Weekend Effects**: Social media performs better on weekends; LinkedIn drops 40%
- **Channel Patterns**: Email has highest CVR; Google Ads highest volume

### Customer Data
- **Segment Profiles**: Premium customers have 2.5x higher LTV
- **Churn Indicators**: Low satisfaction + high support tickets = churn risk
- **Age-Income Correlation**: Peak income at age 45-50

### Product Sales
- **Category Performance**: Electronics highest volume; Fashion highest margins
- **Regional Variations**: West and South regions outperform
- **Quarterly Patterns**: Q4 electronics surge; Q2-Q3 fashion surge

### ML Model (Lead Scoring)
- **AUC**: ~0.75-0.80 (good predictive performance)
- **Key Features**: Webinar attendance and form submissions are strongest predictors
- **Learning Curve**: Model is well-calibrated, slight variance remains

---

## 📊 Visualization Mapping

| Chart Type | Data Source | Key Columns |
|------------|-------------|-------------|
| Bar Chart | campaign_performance | channel, revenue |
| Grouped Bar | campaign_performance | region, quarter, revenue |
| Stacked Bar | campaign_performance | month, campaign_type, spend |
| Line Chart | campaign_performance | date, revenue |
| Area Chart | campaign_performance | date, channel, conversions |
| Histogram | customer_data | age |
| Box Plot | customer_data | customer_segment, lifetime_value |
| Violin Plot | customer_data | nps_category, satisfaction_score |
| Scatter Plot | customer_data | income, lifetime_value, customer_segment |
| Bubble Chart | campaign_performance (agg) | ctr, conversion_rate, spend |
| Heatmap | correlation_matrix | all columns |
| Calendar Heatmap | campaign_performance | date, revenue |
| Pie/Donut | channel_attribution | channel, model columns |
| Treemap | product_sales | category, subcategory, product_name, sales |
| Sunburst | customer_data | region, city_tier, customer_segment |
| Funnel | funnel_data | stage, visitors |
| Choropleth | geographic_data | state, latitude, longitude, revenue |
| Bubble Map | geographic_data | latitude, longitude, store_count, satisfaction |
| Confusion Matrix | lead_scoring_results | actual_converted, predicted_class |
| ROC Curve | lead_scoring_results | actual_converted, predicted_probability |
| Learning Curve | learning_curve | training_size, train_score, validation_score |
| Feature Importance | feature_importance | feature, importance |

---

## 🎯 Assignment Deliverables

1. **Streamlit Dashboard** - All 20+ visualizations
2. **Source Code** - Well-documented Python files
3. **Insights Report** - 2-page business insights summary
4. **Presentation** - 5-minute video walkthrough

---

## 📧 Questions?

Contact your course instructor.

**Good luck! Let data tell the story.** 📊✨
