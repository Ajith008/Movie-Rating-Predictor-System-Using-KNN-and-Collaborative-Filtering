import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from math import sqrt

# This function to load datasets
def load_datasets():
    movies_df = pd.read_csv('/Users/Manish/Downloads/CS504-Project/movies.csv')
    ratings_df = pd.read_csv('/Users/Manish/Downloads/CS504-Project/ratings.csv')
    return movies_df, ratings_df

# This function merges the dataset based on 'movieId'
def merge_datasets(movies_df, ratings_df):
    return pd.merge(ratings_df, movies_df, on='movieId')

def predict_and_calculate_rmse(train_df, test_df, k_value):
    # Extract features and target
    features_train = train_df[['userId', 'movieId']].to_numpy()
    target_train = train_df['rating'].to_numpy()
    features_test = test_df[['userId', 'movieId']].to_numpy()
    
    predictions = []
    
    for test_point in features_test:
        # Calculate distances to all training points
        distances = euclidean_distances([test_point], features_train)[0]
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_ratings = target_train[sorted_indices]
        
        # Select top k neighbors
        k_distances = sorted_distances[:k_value]
        k_ratings = sorted_ratings[:k_value]
        
        # Calculate weights based on the given formula
        d1 = k_distances[0]  # distance to the 1st nearest neighbor
        dk = k_distances[-1]  # distance to the k-th nearest neighbor
        weights = []
        for i in range(k_value):
            if i == 0:
                weights.append(1)  # weight for the 1st nearest neighbor is 1
            else:
                wi = (dk - k_distances[i]) / (dk - d1) if dk != d1 else 1  # Handle division by zero
                weights.append(wi)
        
        # Calculate weighted average prediction
        weighted_sum = np.sum([rating * weight for rating, weight in zip(k_ratings, weights)])
        sum_weights = np.sum(weights)
        prediction = weighted_sum / sum_weights if sum_weights != 0 else np.mean(k_ratings)
        predictions.append(prediction)
    
    # Add predictions to the test DataFrame
    rounded_predictions = np.round(predictions, 2)
    test_df[f'predicted_rating_k{k_value}'] = rounded_predictions
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_df['rating'], test_df[f'predicted_rating_k{k_value}']))
    return rmse

# This function predicts and saves predicted ratings to a csv file for specified k
def predict_and_save_to_csv(train_df, test_df, k_values):
    features_train = train_df[['userId', 'movieId']].to_numpy()
    target_train = train_df['rating'].to_numpy()
    features_test = test_df[['userId', 'movieId']].to_numpy()

    for k in k_values:
        predictions = []
        
        for test_point in features_test:
            # Calculate distances to all training points
            distances = euclidean_distances([test_point], features_train)[0]
            sorted_indices = np.argsort(distances)
            sorted_distances = distances[sorted_indices]
            sorted_ratings = target_train[sorted_indices]
            
            # Select top k neighbors
            k_distances = sorted_distances[:k]
            k_ratings = sorted_ratings[:k]
            
            # Calculate weights based on the given formula
            d1 = k_distances[0]  # distance to the 1st nearest neighbor
            dk = k_distances[-1]  # distance to the k-th nearest neighbor
            weights = []
            for i in range(k):
                if i == 0:
                    weights.append(1)  # weight for the 1st nearest neighbor is 1
                else:
                    wi = (dk - k_distances[i]) / (dk - d1) if dk != d1 else 1  # Handle division by zero
                    weights.append(wi)
            
            # Calculate weighted average prediction
            weighted_sum = np.sum([rating * weight for rating, weight in zip(k_ratings, weights)])
            sum_weights = np.sum(weights)
            prediction = weighted_sum / sum_weights if sum_weights != 0 else np.mean(k_ratings)
            predictions.append(prediction)
        
        # Round predictions and calculate RMSE
        rounded_predictions = np.round(predictions, 2)
        test_df[f'predicted_rating_k{k}'] = rounded_predictions
        rmse = sqrt(mean_squared_error(test_df['rating'], rounded_predictions))
        
        # Save results to CSV
        output_file = f'/Users/Manish/Downloads/CS504-Project/output/ratings_predictions_10_knn_wt_k{k}.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output directory exists
        test_df.to_csv(output_file, columns=['userId', 'movieId', 'rating', f'predicted_rating_k{k}'], index=False)
        print(f"Saved predictions to {output_file} with RMSE: {rmse:.2f}")

# This function loads data and merging datasets
movies_df, ratings_df = load_datasets()
merged_df = merge_datasets(movies_df, ratings_df)

# Splitting the merged dataset in train split and testing split
train_df, test_df = train_test_split(merged_df, test_size=0.1, random_state=42)

# Predicting and saving predicted ratings to CSV for k=3, k=5, and k=10
predict_and_save_to_csv(train_df, test_df.copy(), [3, 5, 10])


#
import matplotlib.pyplot as plt
rmse_k3 = predict_and_calculate_rmse(train_df, test_df, 3)
rmse_k5 = predict_and_calculate_rmse(train_df, test_df, 5)
rmse_k10 = predict_and_calculate_rmse(train_df, test_df, 10)
k_values = [3, 5, 10]
rmse_values = [rmse_k3, rmse_k5, rmse_k10]

# Create a bar graph with different colors for each bar
colors = ['b', 'g', 'r']

# Create a bar graph with equal space between bars on the x-axis
plt.figure(figsize=(8, 6))
plt.bar(range(len(k_values)), rmse_values, color=colors, align='center')
plt.title('RMSE for Different Numbers of Neighbors (k) (Weighted Knn)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.xticks(range(len(k_values)), k_values)
plt.grid(axis='y')

# Add RMSE values on top of each bar
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()