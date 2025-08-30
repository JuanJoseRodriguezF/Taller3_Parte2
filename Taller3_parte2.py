# %% [markdown]
# ## Data Pipelines - Prefect - Example and Workshop
# ### Big Data Tools 
# #### M.Sc. in Applied Analytics (coterminal course)
# Fac. de Ingeniería -  Universidad de la Sabana<br>
# Prof.: Hugo Franco, Ph.D.

# %%
import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prefect import flow, task
from kaggle import KaggleApi
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from functools import wraps

# ---------------- CONFIG ----------------
LOCAL_KAGGLE_DIR = './auth'
DOWNLOAD_DIRECTORY = './data'
TARGET_DATASET = 'sushilyeotiwad/wheat-seed-dataset'
POSTGRES_CONN = "postgresql://psqluser:psqlpass@localhost:5433/bigdatatools1"

# Decorador para medir tiempos
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end = time.time()
            print(f"Task {func.__name__} took {end - start:.2f} seconds")
    return wrapper

def get_directory_size(path='.'):
    """Return folder size in bytes"""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def human_readable_size(size_bytes):
    """Convert bytes into human readable format"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def size_report_decorator(func):
    """Decorator to report download folder size before and after task"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = get_directory_size(DOWNLOAD_DIRECTORY)
        print(f"[Before] Folder size: {human_readable_size(before)}")
        result = func(*args, **kwargs)
        after = get_directory_size(DOWNLOAD_DIRECTORY)
        print(f"[After] Folder size: {human_readable_size(after)}")
        return result
    return wrapper

# ---------------- TASKS -----------------

@task(name="Extract Data", retries=3, retry_delay_seconds=10)
@timing_decorator
@size_report_decorator
def extract_data() -> pd.DataFrame:
    try:
        if not os.path.exists(DOWNLOAD_DIRECTORY):
            print(f"Creating directory: {DOWNLOAD_DIRECTORY}")
            os.makedirs(DOWNLOAD_DIRECTORY)

        api = KaggleApi()
        api.authenticate()

        print(f"Downloading dataset '{TARGET_DATASET}'...")
        api.dataset_download_files(TARGET_DATASET, path=DOWNLOAD_DIRECTORY, unzip=True)

        csv_path = os.path.join(DOWNLOAD_DIRECTORY, "seeds_dataset.csv")
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded with shape {df.shape}")
        return df

    except Exception as e:
        raise RuntimeError(f"Error in extract_data: {e}")


@task(name="Transform Data")
@timing_decorator
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_clustering = df.drop('Class_(1, 2, 3)', axis=1)

        # --- Correlation heatmap ---
        correlation_matrix = df_clustering.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Seed Features')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()

        # --- Elbow method ---
        X = df_clustering[['Asymmetry_coefficient', 'Length_of_kernel_groove']]
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.savefig('elbow_method.png')
        plt.close()

        # --- Best model (K=3) ---
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_clustering['cluster'] = kmeans.fit_predict(X)

        # --- Scatter plot with centroids ---
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_clustering['Asymmetry_coefficient'],
                              df_clustering['Length_of_kernel_groove'],
                              c=df_clustering['cluster'], cmap='viridis')
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
        plt.title('K-means Clustering of Seeds (K=3)')
        plt.xlabel('Asymmetry_coefficient')
        plt.ylabel('Length_of_kernel_groove')
        plt.legend()
        plt.grid(True)
        plt.savefig('kmeans_clustering_plot.png')
        plt.close()

        print("Transformation complete, clusters and plots generated")
        return df_clustering

    except Exception as e:
        raise RuntimeError(f"Error in transform_data: {e}")


@task(name="Load Data")
@timing_decorator
def load_data(df: pd.DataFrame):
    engine = None
    try:
        engine = create_engine(POSTGRES_CONN)
        table_name = "seed_clusters"
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data loaded successfully into table '{table_name}'")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    finally:
        if engine is not None:
            engine.dispose()

# ---------------- FLOW -----------------

@flow(name="ETL Pipeline - Seeds Dataset")
def etl_pipeline():
    try:
        df_raw = extract_data()
        df_transformed = transform_data(df_raw)
        load_data(df_transformed)
    except Exception as e:
        print(f"Pipeline failed: {e}")

# %%
# ---------------- MAIN -----------------

if __name__ == "__main__":
    etl_pipeline()

# %% [markdown]
# __Challenges (Workshop 3, part 2):__
# 1. Organize the code of the example and complete the data pipeline: 
#     - Use `try-except-finally` blocks as required 
#     - Every individual process must be wrapped as a `task` using the corresponding decorator and its parameters when necessary. 
#     - Create a `transform_data` task focused on data cleansing
#     - Create a `load_data` task to create a table in a Dockerized PostgreSQL database and populate with with the clustering-oriented Dataframe
#     - Invoke the tasks in the proper order in the `flow`
#     - Use the `timing_decorator` to report the duration of each task 
# 1. Using the function get_directory_size, create a decorator to get and report the size of the downloaded dataset (size in bytes of the download folder)
# 1. Report the size in the previous question in a human readable unit
# 1. Use the attribute kmeans.cluster_centers_ and scatter plot to add the centroids of the best KMeans model (best K parameter) to the scatter plot
#     - `centroids = kmeans.cluster_centers_`
# 


