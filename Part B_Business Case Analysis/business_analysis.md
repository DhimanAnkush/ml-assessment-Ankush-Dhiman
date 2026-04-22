# B1. Problem Formulation

**(a) Machine Learning Formulation**

To address the retailer's objective, we formulate this as a Supervised Learning - Regression problem.

**Target Variable:** items_sold (The count of products sold per store, per month).

**Candidate Input Features:**

**Store attributes:** store_id, location_type, store_size, competition_density.

**Promotion attributes:** promotion_type.

Temporal/Contextual features: month, is_weekend, is_festival, monthly_footfall.

**Justification** of Problem Type: While the end goal is to "choose" a promotion (which sounds like classification), the underlying mechanic is predicting a continuous quantity (items_sold) for each possible promotion scenario. By predicting volume, the business can rank the five promotions and select the one with the highest expected output. Regression provides more granular insights into the expected lift compared to a simple classification label.

**(b) Target Variable Reliability:** **Volume vs. Revenue**
Using items sold (volume) is more reliable than revenue because revenue is heavily skewed by product price points. For example, a "Category-Specific Offer" on expensive winter coats might generate high revenue with very few items sold, whereas a "BOGO" on socks might move massive volume with lower revenue.

**Reliability:** Volume directly reflects customer engagement and the "clearing" of inventory, which is the primary goal of a promotion. Revenue can fluctuate due to price changes or inflation, masking the true effectiveness of the marketing tactic.

**Broader Principle:** This illustrates the principle of Proximal vs. Distal targets. In ML, the target variable should be the one most directly influenced by the action (the promotion) and least affected by external "noise" (like varying price tags or high-value outliers).

**(c) Alternative Modelling Strategy**
Instead of a single global model, I propose a Clustered (or Segmented) Modelling approach. We can group the 50 stores into three clusters based on location_type (Urban, Semi-Urban, Rural).

**Justification:** Customer behavior in an Urban flagship store (high footfall, high competition) is fundamentally different from a Rural store. A global model might "average out" these nuances, leading to poor recommendations for both. By training segment-specific models or using a Hierarchical/Mixed-Effects model, we allow the algorithm to learn specific coefficients for location-based sensitivities while still sharing some global patterns.


# B2. Data and EDA Strategy

**(a) Data Integration and Grain**
The tables would be joined using a series of "Left Joins" starting with the transactions table as the base.

**Join Chain:** Transactions $\rightarrow$ Promotion Details (on promo_id) $\rightarrow$ Store Attributes (on store_id) $\rightarrow$ Calendar (on date).

**Aggregation:** Raw transaction-level data must be aggregated to a Store-Month grain. We would sum the items_sold and average the footfall and competition_density for that period.

**Final Grain:** One row = One specific Store in one specific Month (e.g., Store 12, March 2024).

**(b) Exploratory Data Analysis (EDA)**

**Promotion vs. Volume Boxplot:** I would look for the variance and median items sold for each of the five promotion types. This reveals which promos are generally high-performers and identifies outliers.

**Time-Series Decomposition:** Plotting total items sold over the three years to identify seasonality (e.g., holiday peaks) and long-term trends. This influences whether we need to add "month" or "lagged sales" as features.

**Correlation Heatmap:** Examining the relationship between competition_density, footfall, and items_sold. If footfall and volume are 0.99 correlated, we might have a redundancy issue (multicollinearity).

**Location-Promo Interaction Plot:** A bar chart showing average sales for each promo, grouped by location_type. If "Free Gift" works in Urban stores but fails in Rural ones, it confirms the need for location-based features or segmented models.

**(c) Handling Promotion Imbalance**

The 80% non-promotion bias is a "Natural Baseline" issue. If the model is trained on this, it may learn to predict "No Promotion" as the safest bet.

**Effect:** The model may struggle to distinguish the subtle "lift" caused by specific promotions because the "No Promo" signal is too dominant.

**Steps to Address:** I would use stratified sampling to ensure the 20% of promotional data is well-represented in training. Alternatively, I would create a "Baseline Lift" feature—calculating the difference between items sold during a promo and the average items sold in that same store during "No Promo" months—effectively making the model predict the increase in sales rather than the raw total.


# B3. Model Evaluation and Deployment

**(a) Split and Evaluation**

**Split Strategy:** A Temporal (Time-Series) Split is required. We would use the first 30 months for training and the most recent 6 months for testing.

**Why Random is Inappropriate:** Randomly shuffling months would cause look-ahead bias. The model would "know" that sales were high in December 2025 while trying to predict sales for January 2024. In the real world, we only have the past to predict the future.

**Metrics:**

**MAE (Mean Absolute Error):** Interpreted as "How many items off are we on average?" (Easier for business stakeholders to understand).

**RMSE (Root Mean Squared Error):** Useful for penalizing large misses, ensuring we don't have a catastrophic inventory shortage in a major store.

**(b) Feature Importance and Communication**
To explain why Store 12 gets different recommendations in December vs. March, I would use SHAP (SHapley Additive exPlanations) values.

**The Investigation:** In December, the model likely gives high weight to the is_festival and month_12 features. SHAP would show that "Loyalty Points" has a high positive interaction with the "Holiday Shopping" context.

**The Communication:** I would tell the marketing team: "In December, our customers are already in a high-spending holiday mindset, and the model found that Loyalty Points create the best long-term retention lift during this peak. In March, footfall is lower, so the Flat Discount is recommended as a 'price-hook' to drive traffic during a slow month."

**(c) Deployment and Monitoring**
**Deployment:** Save the trained pipeline using joblib or pickle. Wrap it in a REST API (using a tool like Flask or FastAPI).

**Preparation:** At the start of a month, a script pulls the latest "Store Attributes" and the upcoming "Calendar" data. It creates five "dummy" rows for each store (one for each promo type).

**Inference:** The model predicts items_sold for all five rows. The row with the highest prediction is sent to the store manager as the recommendation.

**Monitoring:**

**Data Drift:** Monitor if footfall or competition_density values significantly deviate from the training range.

**Concept Drift:** Track the Prediction Error each month. If the MAE starts increasing consistently for three months, it signals that customer buying habits have changed (e.g., due to a new trend or economic shift), triggering an automated retraining pipeline.
