import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Implement your data preprocessing here
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    data = load_data('../data/raw/sample_data.csv')
    processed_data = preprocess_data(data)
    processed_data.to_csv('../data/processed/processed_data.csv', index=False)
