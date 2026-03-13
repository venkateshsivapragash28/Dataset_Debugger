import pandas as pd

def load_data(file_path):
    try:
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}")
        metadata = {
            "num_rows": data.shape[0],
            "num_columns": data.shape[1],
            "column_names": data.columns.tolist(),
            "data_types": data.dtypes.to_dict(),
        }
        return data, metadata
    except Exception as e:
        print(f"Error occurred while loading data from {file_path}: {e}")
        return None
    
if __name__ == "__main__":
    file_path = r"C:\Users\VENKAT\Desktop\DataSets\Automobile_data.csv"
    data, metadata = load_data(file_path)
    if data is not None:
        print("\nDataset Metadata")
        print(metadata)