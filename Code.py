import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

file_path = '/content/drive/MyDrive/MLProj/Cancer_Data.csv'
cancer = pd.read_csv(file_path)

cancer = cancer.drop(columns=['id'], errors='ignore')

# Map the 'diagnosis' column: M -> 1, B -> 0
cancer['diagnosis'] = cancer['diagnosis'].map({'M': 1, 'B': 0})

X = cancer.drop(columns=['diagnosis'])
y = cancer['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def knn_model(X_train, X_test, y_train, y_test, k, normalize=False):
    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    return accuracy, recall, f1

k_values = [5, 10, 15]

# KNN without MinMax Normalization
print("KNN without MinMax Normalization:")
for k in k_values:
    accuracy, recall, f1 = knn_model(X_train, X_test, y_train, y_test, k, normalize=False)
    print(f"K={k}, Accuracy={accuracy:.5f}, Recall={recall:.5f}, F1-Score={f1:.5f}")

# KNN with MinMax Normalization
print("\nKNN with MinMax Normalization:")
for k in k_values:
    accuracy, recall, f1 = knn_model(X_train, X_test, y_train, y_test, k, normalize=True)
    print(f"K={k}, Accuracy={accuracy:.5f}, Recall={recall:.5f}, F1-Score={f1:.5f}")