# Distributed Job Recommendation Engine (PySpark on AWS EMR)

A job recommender built on real job-application data. It learns semantic embeddings for job postings and user work histories with Word2Vec, retrieves candidate jobs with locality-sensitive hashing, and scores user–job pairs with a neural network. Everything runs as Spark jobs on AWS EMR, with data and intermediate artifacts stored in S3.

![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![AWS](https://img.shields.io/badge/AWS%20EMR%20%2B%20S3-232F3E?style=flat-square&logo=amazonwebservices&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

## Data

CareerBuilder job-application data from the Kaggle *Job Recommendation Challenge*: job postings, user profiles, user work histories, and applications, partitioned into 13 time windows (3 GB+ raw). The profile table alone covers ~390K users.

## Pipeline

```
jobs (per window) ──► strip HTML, tokenize ──► Word2Vec (100-d) ──► job vectors ─────────────┐
user_history      ──► job-title tokens     ──► Word2Vec (100-d) ──► mean ──► history vectors │
users             ──► EDA, StringIndexer (state, major) + history vectors ──► user features  │
apps              ──► join to job vectors per window ─────────────────────────────────────────┤
                                                                                              ▼
               ┌──────────── Retrieval: BucketedRandomProjectionLSH over job vectors ──► top-k jobs per user
               └──────────── Ranking:   MLP on [user vector ‖ job vector] ──► P(apply)
```

| Notebook | Stage |
|---|---|
| `Jobs.ipynb` | Clean job text, train Word2Vec, write per-window job vectors to S3 as Parquet |
| `UserHistory.ipynb` | Embed each user's past job titles and average them into one history vector |
| `UsersEda.ipynb` | Profile EDA and null audit, index categoricals, join history embeddings |
| `Apps.ipynb` | Join applications to job and user vectors for a window |
| `KNN.ipynb` | Represent each user as the mean of their applied-job vectors, then retrieve nearest jobs with LSH |
| `Nueral_Net_Prep.ipynb` | Build training pairs: positives = applications, negatives = 3 sampled non-applied jobs per positive |
| `Neural_Network.ipynb` | Train and evaluate the ranking MLP |

## Models

**Retrieval (LSH-KNN).** Spark ML `BucketedRandomProjectionLSH` (3 hash tables) returns approximate nearest neighbors in embedding space, avoiding a full user × job distance matrix.

**Ranking (MLP).** Dense 256 → 128 → 64 → sigmoid with 0.3 dropout, trained with binary cross-entropy on concatenated user and job embeddings.

## Results

| Model | Split | Accuracy | ROC-AUC |
|---|---|---|---|
| Ranking MLP | 20% held-out (window 6 sample) | 95.5% | 0.997 |

**Caveats.** These numbers come from a small sampled subset of one window. The negatives are random non-applied jobs, which makes the task easier than ranking against the jobs a user actually considered. Treat them as proof that the pipeline works end to end, not as a production-level estimate.

## Known issues & next steps

- **Retrieval evaluation bug.** The precision@k calculation in `KNN.ipynb` labels a recommendation as a hit by checking whether `JobID` is non-null after a left join. `JobID` is the join key, so it is always non-null and precision comes out as exactly 1.0. The fix is to test a column that exists only on the applications side.
- Evaluate on a later window than the one used for training (temporal split) with ranking metrics (precision@k, NDCG)
- Sample hard negatives (jobs in the same city or category) instead of random ones
- Replace averaged Word2Vec with a sentence-embedding model for job descriptions

## Running

The notebooks target an EMR Studio cluster with Spark preconfigured. Replace the `s3://…` paths with your own bucket, upload the Kaggle TSVs, and run the notebooks in the order listed in the table above.
