import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

def load_data(file_path):
    """Loads dataset from a file path."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data: Handle categorical variables, feature selection, etc."""
    # Convert categorical variables to dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)

    # Split features and target
    X = data.drop('loan_status', axis=1)  # Features
    y = data['loan_status']  # Target

    return X, y

def train_model(X, y):
    """Train the Random Forest Classifier model."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    model = RandomForestClassifier(random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    return model

if __name__ == "__main__":
    # Load the data
    file_path = r"C:\Users\Windows\Desktop\mlops\loan_approval\artifacts\credit_data.csv"
    data = load_data(file_path)

    # Preprocess the data
    X, y = preprocess_data(data)

    # Train the model
    trained_model = train_model(X, y)

    # Save the model to disk
    dump(trained_model, 'model.joblib')
