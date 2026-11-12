# Zomato_Restaurant_Clustering_Unsuupervised_AI-ML_Model
The Zomato Restaurant Clustering Unsupervised AI-ML Model is an advanced analytics project designed to segment the restaurant landscape by analyzing features such as cuisine, cost for two, location, and user ratings using unsupervised learning technique.
# Zomato Restaurant Clustering

An unsupervised machine learning pipeline designed to segment the Indian restaurant market using metadata and customer sentiment.

## Overview

This project uses Zomato data to uncover patterns in the Indian restaurant industry. It solves the problem of unstructured restaurant data by grouping dining options into logical business segments and analyzing customer sentiment to gauge satisfaction. The insights help customers find suitable restaurants and assist Zomato in identifying thriving segments and market gaps.

## Key Features & Pipeline Steps

### 1. Data Understanding & Wrangling

* **Dataset:** Merged restaurant metadata (105 rows, 6 columns) and reviews (10,000 rows, 7 columns) into a single dataset of 10,000 records and 12 columns.


* **Cleaning:** Removed 36 duplicate rows.


* **Imputation:** Resolved 5,000 missing values in 'Collections' using a KNNImputer. Applied mode imputation for categorical columns ('Timings', 'Reviewer', 'Metadata', 'Time') and median imputation for numerical columns ('Rating', 'Pictures'). Filled empty 'Review' values with an empty string.


* **Outlier Treatment:** Applied Winsorization at the 1st and 99th percentiles to numerical columns ('Cost', 'Pictures', 'Review_Length') to mitigate bias without data loss.



### 2. Exploratory Data Analysis (EDA)

* **Ratings:** Distribution shows extreme spikes at 1.0 and 5.0, indicating customers review primarily for extreme experiences.


* **Cost:** The market is highly concentrated in the ₹400-600 range, with a distinct gap around ₹1200 separating everyday and premium dining.


* **Cuisines:** North Indian and Chinese cuisines completely dominate the market.


* **Engagement:** A strong positive correlation (0.47) exists between the number of uploaded pictures and review length.


* **Pricing vs. Quality:** A very weak correlation (0.14) between Cost and Rating proves expensive restaurants do not guarantee better experiences.



### 3. Feature Engineering & Preprocessing

* **Categorical Encoding:** Applied Label Encoding to high-cardinality features and Multi-Label One-Hot Encoding to 'Collections' and 'Cuisines'.


* **NLP Pipeline:** Expanded slangs, converted to lowercase, removed punctuations, URLs, and stopwords. Applied Lemmatization using WordNetLemmatizer and extracted features using TF-IDF Vectorization (max 1000 features).


* **Dimensionality Reduction:** Dropped noise using a Variance Threshold (reducing features to 67) and optimized scaled numeric features using Principal Component Analysis (PCA) retaining 100% variance across 4 components.



### 4. Clustering & Modeling

* **K-Means (Selected Model):** Tuned via exhaustive ParameterGrid search (init='random', n_init=20) to find the optimal configuration ($k=2$). It achieved a Silhouette Score of 0.1764 and was selected for its hard cluster assignments and scalability.


* **Hierarchical Clustering:** Evaluated for baseline visual comparison using a Dendrogram to map market relationships. Showed optimal results with 'average' linkage.


* **DBSCAN:** Tested but rejected due to poor performance on uneven density data, mislabeling 44.8% of the dataset as noise.



### 5. Deployment

* Serialized the final K-Means model, Hierarchical model, and StandardScaler using the Pickle library for production deployment.

## Business Value

* **Targeted Marketing:** The separation of premium and budget clusters allows Zomato to run highly targeted marketing campaigns, improving ROI.


* **Recommendation Optimization:** Eliminates mismatched recommendations (e.g., suggesting ₹2000 fine dining to budget users) by matching users strictly within their cluster tier.


* **Trust and Safety:** Using noise ratios allows the platform to flag anomaly restaurants (overpriced with terrible ratings) for manual audit or delisting.
