# IBM-Article-Recommendation-System

## Project Overview
The goal is to recommend relevant IBM Watson Studio community articles to users by exploring and combining:
1) **Popularity (rank-based)**  
2) **User–user collaborative filtering (CF)**  
3) **Content-based** (TF-IDF on titles + LSA + KMeans)  
4) **Matrix factorization** (Truncated SVD on the interaction matrix)

---

## Dataset
A single CSV of implicit interactions (each row = user viewed an article). Expected columns:

- `user_id` *(or `email`, which is mapped to `user_id`)*  
- `article_id`  
- `title` *(optional; used by content-based. If missing, the code synthesizes “Article {id}”.)*

**Notes**
- No explicit ratings; viewing an article counts as positive feedback (1).
- For Udacity submission, **do not** include the dataset in your ZIP; the workspace provides it at `data/user-item-interactions.csv`.

---

## Methods

### Rank-Based (Popularity)
Recommends globally most-viewed articles; used for cold-start and as a stable fallback.

### User–User Collaborative Filtering
- Builds a binary **user×item** matrix.
- Similarity = dot product over binary vectors (overlap of seen items).
- **Improved version** prefers neighbors with more total interactions and ranks candidate items by global popularity.

### Content-Based (Titles → TF-IDF + LSA + KMeans)
- Vectorizes titles with **TF-IDF**; reduces with **TruncatedSVD (LSA)**; clusters with **KMeans**.
- For a query article, fetch others in the same cluster and rank by **unique-user popularity**.

### Matrix Factorization (SVD)
- Fits **TruncatedSVD** on the user×item matrix.
- Picks latent dimension by curve shape (and ~90% EVR heuristic).
- Item–item similarity computed in latent space via cosine similarity.

### Evaluation
- **Hit-Rate@10** on a simple train/test split (robust to small datasets; stratifies when feasible).

---

## Results

**Exploratory Data Analysis**
- Median interactions per user: **3**  
- Total interactions: **45,993**  
- Max interactions by a single user: **364**  
- Most viewed article id: **1429.0** with **937** views  
- Unique articles with ≥1 interaction: **714**  
- Unique users: **5,148**  
- Total unique articles: **714**

**Top-10 Popular Titles**
- use deep learning for image classification  
- insights from new york car accident reports  
- visualize car data with brunel  
- use xgboost, scikit-learn & ibm watson machine learning apis  
- predicting churn with the spss random tree algorithm  
- timeseries data analysis of iot events by using jupyter notebook  
- healthcare python streaming application demo  
- finding optimal locations of new store using decision optimization  
- apache spark lab, part 1: basic concepts  
- finding optimal location of new store using decision optimization

**Content-Based**
- Chosen clusters (silhouette-guided): **K = 29**  
- Example similar-to `0.0`: `[730.0, 470.0, 651.0, 382.0, 103.0]`

**SVD**
- Chosen latent features (≈90% EVR): **k = 50**  
- Example latent similar-to `0.0`: `[1112.0, 1124.0, 1292.0, 1066.0, 409.0]`

**Offline Evaluation**
- **Hit-Rate@10**: **0.259** (users evaluated: **1,054**; split: non-stratified fallback)

---

## Getting Started

### Colab Quick Start
1. Upload:
   - `Recommendations_with_IBM.ipynb`
   - `project_tests.py`
   - `user-item-interactions.csv`
2. Add this small setup cell **at the top** (already included in my run) to satisfy Udacity’s expected path:
   - Copy `user-item-interactions.csv` → `data/user-item-interactions.csv`.
3. Run all cells top→bottom. Inline tests should print “passed” messages.
