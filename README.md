# SmartCart Clustering System

SmartCart is an e-commerce platform with a growing customer base — and right now, it treats every customer the same way. This project fixes that by using clustering to figure out who SmartCart's customers actually are, so marketing can stop being one-size-fits-all.

---

##  The Problem

SmartCart has collected data on **2,240 customers**, covering **22 attributes** — demographics, how much people spend, how they shop (web, catalog, in-store), and whether they respond to campaigns.

The issue is that all of this data isn't being used to actually *understand* customers. Everyone gets the same marketing, regardless of whether they're a high-spending loyal customer or someone who barely engages. That means:

- Marketing budget gets wasted on people it won't work on
- High-value customers aren't being specifically retained
- Customers who are about to churn aren't being caught in time

### What I set out to do

Use unsupervised machine learning to group SmartCart's customers into a handful of clear, meaningful segments based on how they spend, how engaged they are, and how loyal they've been — so the marketing team can actually target the right people with the right message.

---

##  The Data

Each row is one customer. Here's what's in there:

### 1. Customer Demographics
| Feature | Description |
|---|---|
| `ID` | Unique customer identifier |
| `Year_Birth` | Year of birth of the customer |
| `Education` | Highest education level achieved |
| `Marital_Status` | Marital status of the customer |
| `Income` | Yearly household income |
| `Kidhome` | Number of small children in household |
| `Teenhome` | Number of teenagers in household |
| `Dt_Customer` | Date when customer enrolled with SmartCart |

### 2. Purchase Behaviour (Amount Spent)
| Feature | Description |
|---|---|
| `MntWines` | Amount spent on wine products |
| `MntFruits` | Amount spent on fruits |
| `MntMeatProducts` | Amount spent on meat products |
| `MntFishProducts` | Amount spent on fish products |
| `MntSweetProducts` | Amount spent on sweet products |
| `MntGoldProds` | Amount spent on gold products |

### 3. Purchase Behaviour (Frequency)
| Feature | Description |
|---|---|
| `NumDealsPurchases` | Purchases made using discounts |
| `NumWebPurchases` | Purchases made through website |
| `NumCatalogPurchases` | Purchases made through catalog |
| `NumStorePurchases` | Purchases made in physical stores |
| `NumWebVisitsMonth` | Number of visits to website per month |

### 4. Customer Feedback & Constants
| Feature | Description |
|---|---|
| `Recency` | Number of days since last purchase |
| `Complain` | Customer complained in last 2 years (1 = Yes, 0 = No) |

---

##  What I used

- **Python**, with pandas/numpy for data wrangling and matplotlib/seaborn for plots
- **scikit-learn** for pretty much everything else: `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, and `silhouette_score`
- **kneed** — small library that auto-detects the "elbow" point on a curve, instead of eyeballing it

---

##  How I approached it

**1. Loaded the data and had a look around**
Read in `smartcart_customers.csv`, checked the shape, checked for missing values.

**2. Cleaned up what was missing**
`Income` had some gaps, filled with the median.

**3. Engineered some better features**
The raw columns weren't quite ready for clustering, so I reshaped a few things:
- Turned `Year_Birth` into `Age`
- Parsed `Dt_Customer` into an actual date, then worked out `Customer_Tenure_Days` — how long each customer's been with SmartCart, relative to whoever joined most recently
- Added up all six spending columns (wine, fruit, meat, fish, sweets, gold) into one `Total_spendings` number
- Combined `Kidhome` and `Teenhome` into a single `Total_Children`
- Simplified `Education` down into three sensible buckets: UnderGraduate, Graduate, Postgraduate
- Simplified `Marital_Status` into just Married or Single (there were some odd values in there — "Absurd", "YOLO" — which I folded into Single)

**4. Dropped what I no longer needed**
Once the new features existed, the old raw columns (`ID`, `Year_Birth`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Marital_Status`, and the individual spending columns) weren't doing anything useful anymore, so I dropped them.

**5. Went looking for outliers**
A pairplot across `Income`, `Recency`, `Response`, `Age`, `Total_spendings`, and `Total_Children` made a couple of weird points obvious — a handful of customers with implausible ages and one or two with absurdly high income. Filtered out anyone with `Age >= 90` or `Income >= 600,000`.

**6. Checked how features relate to each other**
A correlation heatmap over the numeric columns — mostly to sanity-check things like Income being tied to spending, and web visits being tied to *not* spending much.

**7. Encoded the categorical stuff**
One-hot encoded `Education` and the new `Relation` column so they could actually be used in clustering.

**8. Scaled everything**
Ran everything through `StandardScaler`, since clustering algorithms are sensitive to features being on wildly different scales.

**9. Reduced dimensions with PCA**
Squeezed everything down to 3 principal components — makes the clustering faster and lets me actually visualize the customer space in 3D.

**10. Figured out how many clusters made sense**
Ran the elbow method (WCSS) across K=1 to 10, and let `kneed` find the elbow automatically instead of guessing. Cross-checked that against the silhouette score across K=2 to 10, then plotted both together to see where they agreed. K=4 came out as the sensible choice.

**11. Actually clustered the customers**
Tried both KMeans and Agglomerative Clustering (Ward linkage), both with 4 clusters, both visualized in 3D over the PCA space. Agglomerative Clustering gave noticeably cleaner separation between groups, so that's what I went with for the final segmentation.

**12. Profiled each cluster**
Looked at cluster sizes, plotted Income against Total Spending colored by cluster, and pulled the mean of every feature per cluster to build a real picture of who's in each group.

---

##  So who are SmartCart's customers, really?

Four fairly distinct groups came out of this:

| Cluster | Who they are | Income | Spending | What stands out |
|---|---|---|---|---|
| **0** | Married families, watching their budget | ~$40k | ~$220 | Most kids on average, visits the site a lot but rarely buys, barely responds to campaigns |
| **1** | Married, well-off, and loyal | ~$73k | ~$1,237 | Buys straight through catalog/store, doesn't browse much, been around the longest |
| **2** | Single, and careful with money | ~$37k | ~$166 | Visits the site more than anyone, but converts the least |
| **3** | Single, high earners, and SmartCart's best audience | ~$71k | ~$1,190 | Buys via catalog/store, and responds to campaigns way more than anyone else (~32%) |

**The pattern here is pretty clean:** income basically decides how much someone spends, and whether someone's married or single splits the four groups almost perfectly. The two wealthier clusters (1 and 3) skip the browsing and buy directly — while the two lower-income clusters (0 and 2) spend a lot of time on the site without much to show for it.

If SmartCart only has budget to focus on one group, it's **Cluster 3** — they already spend well and they're by far the most likely to respond to a campaign.

---

##  What's in this repo

```
smartcart-clustering-system/
│
├── customer_segmentation.ipynb   # everything: cleaning, EDA, PCA, clustering, profiling
├── smartcart_customers.csv       # the raw dataset — 2,240 customers, 22 columns
└── README.md
```

---

##  Running it yourself

```bash
# grab the repo
git clone <repo-url>
cd smartcart-clustering-system

# install what you need
pip install pandas numpy matplotlib seaborn scikit-learn kneed jupyter

# open it up
jupyter notebook customer_segmentation.ipynb
```

---

##  What came out of this

- Went from "everyone gets the same marketing" to **4 distinct, data-backed customer segments**
- Found the segment worth prioritizing (Cluster 3) — high spenders who actually respond to campaigns
- Noticed a pattern that's easy to miss otherwise: some customers browse a lot without buying, while others barely browse and just buy — that's a genuinely useful thing to know when deciding where to spend marketing effort.