import pandas as pd

# Load the provided CSV files
links_path = '/Users/Manish/Downloads/CS504-Project/links.csv'
movies_path = '/Users/Manish/Downloads/CS504-Project/movies.csv'
ratings_path = '/Users/Manish/Downloads/CS504-Project/ratings.csv'
tags_path = '/Users/Manish/Downloads/CS504-Project/tags.csv'

links_df = pd.read_csv(links_path)
movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)
tags_df = pd.read_csv(tags_path)

# Display the first few rows of each dataframe to verify their structure
links_df.head(), movies_df.head(), ratings_df.head(), tags_df.head()


# Data Cleaning and Preprocessing

# Check for missing values in each dataset
missing_links = links_df.isnull().sum()
missing_movies = movies_df.isnull().sum()
missing_ratings = ratings_df.isnull().sum()
missing_tags = tags_df.isnull().sum()

# Convert timestamps to datetime for readability (ratings and tags datasets)
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
tags_df['timestamp'] = pd.to_datetime(tags_df['timestamp'], unit='s')

# Ensuring consistency in movieId across datasets by checking intersection of movieIds
common_movie_ids = set(links_df['movieId']).intersection(
    movies_df['movieId']
).intersection(
    ratings_df['movieId']
).intersection(
    tags_df['movieId']
)

# Filter datasets to include only common movieIds
links_cleaned = links_df[links_df['movieId'].isin(common_movie_ids)]
movies_cleaned = movies_df[movies_df['movieId'].isin(common_movie_ids)]
ratings_cleaned = ratings_df[ratings_df['movieId'].isin(common_movie_ids)]
tags_cleaned = tags_df[tags_df['movieId'].isin(common_movie_ids)]

# Results of cleaning
{
    "Missing Values": {
        "links": missing_links,
        "movies": missing_movies,
        "ratings": missing_ratings,
        "tags": missing_tags
    },
    "Dataset Sizes": {
        "links_cleaned": len(links_cleaned),
        "movies_cleaned": len(movies_cleaned),
        "ratings_cleaned": len(ratings_cleaned),
        "tags_cleaned": len(tags_cleaned)
    }
}
