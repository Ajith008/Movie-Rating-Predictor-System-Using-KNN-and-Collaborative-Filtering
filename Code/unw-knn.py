#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from math import sqrt

#This function loads the datasets
def load_datasets():
    movies_df = pd.read_csv('/Users/Manish/Downloads/CS504-Project/movies.csv')
    ratings_df = pd.read_csv('/Users/Manish/Downloads/CS504-Project/ratings.csv')
    return movies_df, ratings_df

# This function merges the dataset based on movieId
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
        
        # Expand k to include ties
        cutoff_distance = sorted_distances[k_value - 1]
        expanded_indices = sorted_indices[sorted_distances <= cutoff_distance]
        expanded_ratings = sorted_ratings[:len(expanded_indices)]
        
        # Predict by averaging ratings of all selected neighbors
        prediction = np.mean(expanded_ratings)
        predictions.append(prediction)
    
    # Add predictions to the test DataFrame
    rounded_predictions = np.round(predictions, 2)
    test_df[f'predicted_rating_k{k_value}'] = rounded_predictions
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_df['rating'], test_df[f'predicted_rating_k{k_value}']))
    return rmse



# This function predicts ratings and save it to csv based on k value 
def predict_and_save_to_csv(train_df, test_df, k_values):
    # Extract features and target
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
            
            # Expand k to include ties
            cutoff_distance = sorted_distances[k-1]
            expanded_indices = sorted_indices[sorted_distances <= cutoff_distance]
            expanded_ratings = sorted_ratings[:len(expanded_indices)]
            
            # Predict by averaging ratings of all selected neighbors
            prediction = np.mean(expanded_ratings)
            predictions.append(prediction)
        
        # Round predictions and calculate RMSE
        rounded_predictions = np.round(predictions, 2)
        test_df[f'predicted_rating_k{k}'] = rounded_predictions
        rmse = sqrt(mean_squared_error(test_df['rating'], rounded_predictions))
        
        # Save results to CSV
        output_file = f'/Users/Manish/Downloads/CS504-Project/output/ratings_predictions_10_knn_uw_k{k}.csv'
        test_df.to_csv(output_file, columns=['userId', 'movieId', 'rating', f'predicted_rating_k{k}'], index=False)
        print(f"Saved predictions to {output_file} with RMSE: {rmse:.2f}")
        
# Loading the datasets and merging the datasets
movies_df, ratings_df = load_datasets()
merged_df = merge_datasets(movies_df, ratings_df)

# Splitting the merged datsets into training and testing split
train_df, test_df = train_test_split(merged_df, test_size=0.1, random_state=42)

# Predicting ratings and saving the predicted ratings to csv file when k=3,5 and 10
predict_and_save_to_csv(train_df, test_df.copy(), [3, 5, 10])

#Bar Graph for RMSE vs K
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
plt.title('RMSE for Different Numbers of Neighbors (k) (Unweighted Knn)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.xticks(range(len(k_values)), k_values)
plt.grid(axis='y')

# Add RMSE values on top of each bar
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()


