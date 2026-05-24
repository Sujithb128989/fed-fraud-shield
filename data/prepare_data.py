import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_synthetic_fraud_data(n_samples=10000, n_clients=3):
    """
    Generate synthetic data resembling the Kaggle Credit Card Fraud Detection dataset.
    Features: Time, V1..V28, Amount, Class
    """
    print(f"Generating synthetic dataset with {n_samples} samples...")
    # 28 PCA features + Time + Amount = 30 features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=20,
        n_redundant=2,
        weights=[0.99, 0.01], # Highly imbalanced like fraud data
        flip_y=0.01,
        random_state=42
    )

    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df = pd.DataFrame(X, columns=columns)
    df['Class'] = y

    # Scale Time and Amount to look somewhat realistic
    df['Time'] = np.random.uniform(0, 172800, n_samples)
    df['Amount'] = np.random.lognormal(mean=2, sigma=1.5, size=n_samples)

    return df

def get_kaggle_dataset():
    """
    Download the real Kaggle Credit Card Fraud dataset using Hugging Face datasets.
    """
    try:
        from datasets import load_dataset
        print("Downloading Kaggle Credit Card Fraud dataset from Hugging Face...")
        dataset = load_dataset("David-Egea/Creditcard-fraud-detection")
        df = dataset['train'].to_pandas()
        return df
    except Exception as e:
        print(f"Failed to download from Hugging Face: {e}")
        return None

def prepare_data(n_clients=3):
    os.makedirs('data/clients', exist_ok=True)
    
    csv_path = 'data/creditcard.csv'
    if os.path.exists(csv_path):
        print(f"Loading existing {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        df = get_kaggle_dataset()
        if df is not None:
            print("Saving downloaded dataset to data/creditcard.csv...")
            df.to_csv(csv_path, index=False)
        else:
            df = generate_synthetic_fraud_data(n_samples=30000)

    print(f"Dataset shape: {df.shape}")

    # Split across clients
    # To simulate non-IID data or slight drift, we can sort by some feature or just random split
    # Let's do a simple random split for now, but we can skew it later if needed
    client_dfs = np.array_split(df.sample(frac=1, random_state=42), n_clients)

    for i, client_df_arr in enumerate(client_dfs):
        client_df = pd.DataFrame(client_df_arr, columns=df.columns)
        client_id = f"client_{i+1}"

        # Split into train and test for each client
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42, stratify=client_df['Class'])

        client_dir = f"data/clients/{client_id}"
        os.makedirs(client_dir, exist_ok=True)

        train_df.to_csv(f"{client_dir}/train.csv", index=False)
        test_df.to_csv(f"{client_dir}/test.csv", index=False)
        print(f"Saved data for {client_id}: {len(train_df)} train samples, {len(test_df)} test samples")

if __name__ == "__main__":
    prepare_data(n_clients=3)
