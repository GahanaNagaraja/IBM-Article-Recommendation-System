# IBM-Article-Recommendation-System

## 1) Project Overview
This project builds a recommendation system for IBM Watson Studio community articles.  
It follows the Udacity “Recommendations with IBM” template and expands it with content-based
clustering, matrix factorization, and a small offline evaluation so you can compare methods.

## 2) Objectives
- Explore user–article interaction data and report key stats (per rubric).
- Deliver baseline **rank-based** recommendations for cold-start scenarios.
- Implement **user–user collaborative filtering** (basic and improved).
- Add a **content-based** method using TF-IDF + clustering to recommend similar articles.
- Train an **SVD (matrix factorization)** model for item–item recommendations in latent space.
- Provide a lightweight **offline evaluation** (Hit-Rate@10) and summarize findings.

## 3) Dataset Description
The notebook expects a single CSV of user–article interactions, typically with:
- `user_id` *(or `email`, which is mapped to `user_id`)*  
- `article_id`  
- `title` *(optional but used for content-based; if missing, titles are synthesized as “Article {id}”)*

**Notes**
- No explicit ratings are provided; an interaction implies implicit positive feedback (1).
- In the Udacity workspace, the provided dataset is already available; do **not** include it in your ZIP.

## 4) Technologies Used
- **Python**: data processing & modeling  
- **pandas / numpy**: data wrangling  
- **scikit-learn**: TF-IDF, KMeans, TruncatedSVD, cosine similarity, silhouette score  
- **matplotlib**: simple plots (EVR curve, silhouette vs. K)

## 5) Recommendation Approaches
### 5.1 Rank-Based (Popularity)
- **What**: Recommend the most-viewed articles platform-wide.  
- **Why**: Strong, simple baseline for **new users** (cold start) and tie-breaking.  
- **Key functions**: `get_top_article_ids`, `get_top_articles`, `get_article_names`.

### 5.2 User–User Collaborative Filtering
- **Matrix**: Binary user–item matrix (`1` if user viewed article, else `0`).  
- **Similarity**: Dot-product over binary vectors → more overlap = more similar users.  
- **Basic recs**: Pull items from nearest neighbors that the target user hasn’t seen; rank by global popularity.  
- **Improved recs**: When neighbors tie on similarity, prefer **power users** (more total interactions), then rank candidate items by popularity.  
- **New-user fallback**: Top-N popular articles.  
- **Key functions**: `create_user_item_matrix`, `find_similar_users`, `get_top_sorted_users`,  
  `user_user_recs`, `user_user_recs_part2`, `get_user_articles`.

### 5.3 Content-Based (Titles → TF-IDF + KMeans)
- **Text features**: TF-IDF of article **titles** (works even if only titles are available).  
- **Clustering**: Choose **K** via **silhouette score** (search K ∈ [2, 30], bounded by data size), then fit KMeans.  
- **Recommendation**: For a given article, restrict to its cluster and rank neighbors by cosine similarity in TF-IDF space.  
- **Key function**: `get_similar_articles`.

### 5.4 Matrix Factorization (SVD)
- **Model**: Truncated SVD on the binary user–item matrix (implicit feedback).  
- **k selection**: Plot cumulative Explained Variance Ratio (EVR) and choose the smallest **k** that reaches ~**90% EVR**.  
- **Item–item recommendations**: Compute cosine similarity between item vectors in latent space (`Vtᵀ`).  
- **Exposed matrices**: `U` (users × k), `Sigma` (k,), `Vt` (k × items).  
- **Key function**: `get_svd_similar_article_ids`.


## 6) Results

**Exploratory Data Analysis (from `sol_1_dict`)**
- 50% of individuals have ≤ **3** interactions  
- Total user–article interactions: **45,993**  
- Max interactions by any 1 user: **364**  
- Most viewed article id: **1429.0** with **937** views  
- Unique articles with ≥1 interaction: **714**  
- Unique users: **5,148**  
- Unique articles on platform: **714**

**Rank-Based (Top-10 titles)**
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
- Chosen K (silhouette): **29**  
- Example similar-to `0.0`: `[730.0, 470.0, 651.0, 382.0, 103.0]`

**SVD**
- Chosen latent k (EVR ≈ 0.90): **50**  
- Example item–item (latent) similar to `0.0`: `[1112.0, 1124.0, 1292.0, 1066.0, 409.0]`

**Offline Evaluation**
- **Hit-Rate@10**: **0.259** (users evaluated: **1,054**; split: non-stratified (fallback))


