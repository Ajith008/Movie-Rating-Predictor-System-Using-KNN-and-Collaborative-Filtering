import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

# Function to load datasets
def load_datasets():
    movies_df = pd.read_csv('/Users/Manish/Downloads/CS504-Project/movies.csv')
    ratings_df = pd.read_csv('/Users/Manish/Downloads/CS504-Project/ratings.csv')
    return movies_df, ratings_df

# Function to calculate Ullman similarity
def calculate_similarity_ullman(train_pivot):
    # Center the matrix by subtracting the average rating for each movie
    movie_means = train_pivot.mean(axis=0)  # Average rating per movie
    train_pivot_centered = train_pivot - movie_means

    # Compute standard deviation to identify problematic movies
    std_dev = train_pivot_centered.std(axis=0)
    non_constant_movies = std_dev > 0  # Boolean mask for movies with non-zero variance

    # Filter the centered matrix to include only non-constant movies
    train_pivot_centered_filtered = train_pivot_centered.loc[:, non_constant_movies]

    # Fill NaN values with 0 for correlation calculation
    train_pivot_centered_filled = train_pivot_centered_filtered.fillna(0)

    # Calculate similarity using the Ullman formula (numpy's corrcoef)
    similarity_matrix = np.corrcoef(train_pivot_centered_filled.T)

    # Replace any remaining NaN in the similarity matrix with 0
    similarity_matrix = np.nan_to_num(similarity_matrix)

    return similarity_matrix, movie_means[non_constant_movies]

# Function to predict ratings using Ullman similarity
def predict_ratings_ullman(similarity_matrix, train_pivot, movie_means):
    # Filter the train_pivot to include only non-constant movies
    train_pivot_filtered = train_pivot[movie_means.index]

    # Fill missing values with 0 for weighted sum calculations
    train_pivot_filled = train_pivot_filtered.fillna(0)

    # Compute weighted sum of similarities
    weighted_sum = similarity_matrix.dot(train_pivot_filled.T)

    # Normalize by the sum of absolute similarities (avoid division by zero)
    sum_of_weights = np.abs(similarity_matrix).sum(axis=1, keepdims=True)
    sum_of_weights[sum_of_weights == 0] = np.nan  # Avoid division by zero
    sum_of_weights = np.nan_to_num(sum_of_weights, nan=1)  # Replace NaN with 1 to prevent issues

    # Compute predicted ratings
    predictions = weighted_sum / sum_of_weights

    # Re-add the movie means to predictions
    predictions_df = pd.DataFrame(predictions.T, index=train_pivot.index, columns=movie_means.index)
    predictions_df = predictions_df.add(movie_means, axis=1)  # Add movie means by column alignment

    return predictions_df

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    rmse = sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    return rmse

# Function to perform collaborative filtering using Ullman method and save results
def perform_ullman_collaborative_filtering_and_save(split_ratios):
    movies_df, ratings_df = load_datasets()
    rmse_values = []  # List to store RMSE values
    
    for train_ratio in split_ratios:
        test_ratio = 1 - train_ratio
        train_df, test_df = train_test_split(ratings_df, test_size=test_ratio, random_state=42)
        
        # Create a pivot table for the train set
        train_pivot = train_df.pivot_table(index='userId', columns='movieId', values='rating')
        
        # Calculate similarity and movie means
        similarity_matrix, movie_means = calculate_similarity_ullman(train_pivot)
        
        # Predict ratings using Ullman similarity
        train_predictions = predict_ratings_ullman(similarity_matrix, train_pivot, movie_means)
        
        # Map predictions to test_df
        test_df_filtered = test_df[['userId', 'movieId', 'rating']].copy()
        test_df_filtered['predicted_rating'] = test_df_filtered.apply(
            lambda row: train_predictions.at[row['userId'], row['movieId']] 
            if row['movieId'] in train_predictions.columns and row['userId'] in train_predictions.index
            else 0,  # Assign 0 if no prediction is available
            axis=1
        )
        
        # Round predicted ratings to two decimal places
        test_df_filtered['predicted_rating'] = test_df_filtered['predicted_rating'].round(2)

        # Save test_df with actual and predicted ratings to a CSV file
        test_df_filename = f'/Users/Manish/Downloads/CS504-Project/output/ratings_predictions_{test_ratio*100:.0f}_cf.csv'
        test_df_filtered.to_csv(test_df_filename, index=False)

        # Prepare data for RMSE calculation
        test_pivot = test_df.pivot_table(index='userId', columns='movieId', values='rating')
        test_pivot_masked = test_pivot.where(~test_pivot.isna(), np.nan)
        train_predictions_masked = train_predictions.reindex_like(test_pivot_masked)

        # Compute RMSE
        rmse = calculate_rmse(test_pivot_masked.values.flatten(), train_predictions_masked.values.flatten())
        rmse_values.append(rmse)  # Append RMSE to the list
    
        print(f'RMSE for {train_ratio*100:.0f}% train / {test_ratio*100:.0f}% test split: {rmse:.4f}')
        print(f'Saved test data with actual and predicted ratings to {test_df_filename}')
    
    return rmse_values


split_ratios = [0.6, 0.7, 0.8, 0.9]
rmse_values = perform_ullman_collaborative_filtering_and_save(split_ratios)
