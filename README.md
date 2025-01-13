# Movie-Rating-Predictor-System-Using-KNN-and-Collaborative-Filtering
## Project Description:

This project focuses on developing a movie recommendation system using two popular machine learning techniques: k-Nearest Neighbors (KNN) and Collaborative Filtering (CF). The MovieLens dataset, containing 100,000 ratings for approximately 9,000 movies, was used as the basis for building and evaluating the models. The primary objective was to accurately predict user ratings for movies by analyzing historical data and user preferences.

## Objective:

To build a robust recommender system that can predict movie ratings by leveraging user-item interaction data and evaluating the performance of two distinct algorithms.

Techniques Used:

## k-Nearest Neighbors (KNN)

Supervised learning algorithm widely used in classification and regression tasks.

Non-parametric, meaning it makes no assumptions about the data distribution.

Various values of k (e.g., k=3, 5, 10) and prediction methods (weighted vs. unweighted average) were explored.

## Collaborative Filtering (CF)

One of the core methods in recommender systems, focused on learning user-item interaction patterns.

CF was applied using different training/testing splits to assess its performance.

The method demonstrated high accuracy, especially when sufficient training data was available.

## Dataset:

The MovieLens dataset, containing 100,000 user ratings for 9,000 movies, was utilized. The dataset presented a challenge due to sparsity, where many users had only rated a limited number of movies.

## Evaluation Metrics:

Root Mean Squared Error (RMSE): Used to measure the accuracy of the predicted ratings.

Runtime: Wall-clock time was recorded to compare the computational efficiency of both methods.

## Key Findings:

**k-Nearest Neighbors (KNN)**

Higher values of k generally resulted in better accuracy (lower RMSE) and slightly improved runtime.

Optimal configuration: k = 10 with an unweighted average prediction, achieving an RMSE of 1.04.

**Collaborative Filtering (CF)**

Achieved better accuracy than KNN, with the lowest RMSE of 0.9879 using a 90%-10% training/testing split.

Performance decreased with smaller training sets, emphasizing the need for sufficient data.

## Comparison of Methods:

**Accuracy**: CF outperformed KNN, achieving the lowest RMSE.

**Runtime**: CF had a shorter runtime compared to KNN for larger values of k, making it more efficient for large datasets.

## Conclusion: 

This project successfully demonstrated the development of a movie recommendation system using two distinct algorithms: k-Nearest Neighbors and Collaborative Filtering. While both methods performed well, CF proved to be the superior method in terms of both accuracy and efficiency for the given dataset.

## Key Insights:

KNN is effective for smaller datasets and local associations but struggles with larger, sparser datasets.

Collaborative Filtering excels at capturing latent patterns in user-item interactions but requires a large amount of data for optimal performance.

## Future Work:

1. Implementing hybrid models combining CF and content-based filtering.

2. Exploring deep learning-based recommender systems, such as matrix factorization and neural collaborative filtering.

3. Optimizing runtime further by parallelizing computations and leveraging GPU-based processing.
