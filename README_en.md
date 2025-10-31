# 🧩 Customer Segmentation Analysis

**Dataset:** [https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering)

---

## 1. Data Preparation

Steps performed:

* Converted `Dt_Customer` to datetime:

  ```python
  df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
  ```
* Removed 9 missing values to ensure data consistency.
* Created new features:

  * `Total_Children` — total number of children (`Kidhome + Teenhome`)
  * `Total_Spending` — total customer spending across all categories
  * `Customer_Since` — number of days since registration (`2025 - Dt_Customer`)

---

## 2. Exploratory Data Analysis (EDA)

### Distribution & Boxplots

Plotted normal distributions and boxplots using **Seaborn**.
Identified several patterns and hypotheses:

**Education & Income**

* Customers with higher education levels tend to have higher income.

**Marital Status & Spending**

* People living alone tend to spend less — a useful behavioral feature for segmentation models.

---

### Correlation Matrix Insights

| Variables                           | Correlation | Insight                                                            |
| ----------------------------------- | ----------- | ------------------------------------------------------------------ |
| Total_Spending ↔ NumStorePurchases  | **0.68**    | Strong positive link — main spending comes from offline purchases. |
| Total_Spending ↔ NumWebPurchases    | **0.53**    | Active spenders buy both online and offline.                       |
| NumStorePurchases ↔ NumWebPurchases | **0.52**    | Indicates omnichannel buyer behavior.                              |
| Age ↔ Other Variables               | ~0.11–0.16  | Age is not a key behavioral factor.                                |

**Conclusion:** Active spenders are not tied to a single channel — loyal customers buy everywhere.

---

### Income by Education & Marital Status

| Education  | Marital Status | Average Income |
| ---------- | -------------- | -------------- |
| PhD        | Married        | 79,244         |
| Graduation | Married        | 64,176         |
| PhD        | Together       | 64,176         |

**Findings:**

* Graduation — consistently the leader in income across statuses.
* PhD — stable, high income close to Graduation.
* Basic education — lowest income (2–3× less).

---

## 3. Feature Engineering

Created grouped insights:

```python
group1 = df.groupby("Education")["Total_Spending"].mean().sort_values(ascending=False)
```

Added **TotalAccepted** feature — total accepted marketing offers:

```python
df["TotalAccepted"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]].sum(axis=1)
df["AcceptedAny"] = df["TotalAccepted"] > 0
```

Grouped by marital status:

```python
group2 = df.groupby("Marital_Status")["AcceptedAny"].mean().sort_values(ascending=False)
```

**Insights:**

* Most responsive groups: Together, Married
* Least responsive: Alone, Widow

**Marketing recommendations:**

* Together / Married — family offers, loyalty programs
* Divorced / Single — personal promotions, “for yourself” campaigns
* Alone / Widow — simple offers, basic communication

---

## 4. Age Group Analysis

**Pattern discovered:**

* Income grows almost linearly with age.
* Peak income — 70+, not 40–50 as expected.
* Young (18–29) — lowest income (expected).
* Clients 60+ — most solvent group.

**Possible reasons:** retirement savings, inheritance, or product targeting older audiences.

**Recommendations:**

* For 50+: premium services, loyalty programs, personal offers.
* For 18–39: installments, educational content, youth discounts.

---

## 5. Scaling & Clustering

* All features were scaled to the same range to avoid distorted clusters.
* Used the **Elbow Method** to determine optimal k (via WCSS plot).
* Verified using the **Silhouette Coefficient** → final choice: **k = 4**

Visualized clusters using **PCA** for 2D interpretation.

---

## 6. Cluster Interpretation

### Cluster 0 — “Rich but Inactive”

* Income: 77K | Spending: 1250
* Older (≈58 y/o)
* Prefer offline purchases
* High Recency (70) → long time since last purchase
  **Insight:** Wealthy older clients who became inactive.
  **Action:** Reactivation campaigns (email, bonuses, reminders).

---

### Cluster 1 — “Low-income Inactive”

* Income: 34K | Spending: 103
* Low purchasing activity
* Moderate Recency (49)
  **Insight:** Low purchasing power, browse often but rarely buy.
  **Action:** Discounts, installments, personalized offers.

---

### Cluster 2 — “Loyal High-Value Customers”

* Income: 70K | Spending: 1100
* Active online and offline
* Low Recency (18.9) → recent purchases
  **Insight:** Ideal customers — wealthy, loyal, active.
  **Action:** VIP programs, loyalty rewards, retention focus.

---

### Cluster 3 — “Online-Oriented Mid-Spenders”

* Income: 57K | Spending: 811
* High online activity (7.6 web purchases)
* High Recency (55) → activity decreasing
  **Insight:** Digital-oriented clients who lost interest.
  **Action:** Email remarketing, personalized online campaigns.

---

## 7. Final Summary

* **Cluster 2** — most valuable (high income, active, low Recency).
* **Cluster 0** — wealthy but dormant — reactivation potential.
* **Cluster 3** — online-savvy, worth re-engaging.
* **Cluster 1** — low priority for marketing budget.
