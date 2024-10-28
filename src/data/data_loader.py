import pandas as pd
file_path = r"C:\Users\Windows\Desktop\mlops\loan_approval\artifacts\credit_data.csv"


def load_data(file_path):
    """Loads dataset from a file path."""
    data = pd.read_csv(file_path)
    return data.head()

data = load_data(file_path)
print(data.head()) 