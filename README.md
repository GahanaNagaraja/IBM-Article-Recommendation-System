# IBM-Article-Recommendation-System

## Overview

This project implements a hybrid recommendation system for IBM's article platform using real user interaction data. It demonstrates multiple recommendation strategies, including rank-based, collaborative filtering, content-based filtering, and matrix factorization using SVD. The system is capable of handling various user scenarios—from new users to returning users with deep interaction histories.

---

## Dataset

The dataset provided contains user-article interaction data:
- `user_id` (originally email): unique identifier for each user
- `article_id`: identifier of articles
- `title`: article titles
- Implicit feedback (clicks/views) from users on articles

---

## Features Implemented

### 1. **Exploratory Data Analysis (EDA)**
- Number of unique users and articles
- Most viewed article and its view count
- Median and max number of articles interacted per user

### 2. **Rank-Based Recommendations**
- Returns globally popular articles based on view count
- Useful as a fallback for new users or general homepage recs

### 3. **Collaborative Filtering (User-User)**
- Uses cosine similarity between users in the user-item matrix
- Recommends articles read by similar users

### 4. **Content-Based Filtering**
- Uses TF-IDF vectorization of article titles as proxy for content
- Clusters similar articles using KMeans
- Recommends articles from the same content cluster

### 5. **Matrix Factorization with SVD**
- Reduces dimensionality of the user-item matrix to extract latent features
- Reconstructs predictions for user-article interactions
- Includes both user-based and article-to-article recommendations
- Visualizations for:
  - Explained variance vs. latent features
  - Elbow method for optimal KMeans clustering

### 6. **Cold Start Handling**
- Recommends top-ranked articles for users with no interaction history

---

## Technologies Used

- **Python 3**
- **Pandas**, **NumPy**
- **Scikit-learn** (TF-IDF, KMeans, cosine similarity)
- **Matplotlib** (for elbow and SVD variance plots)

---
## Conclusion
This project demonstrates a robust hybrid recommendation engine capable of delivering personalized and scalable article suggestions by combining rank-based, collaborative, content-based, and matrix factorization techniques.
