import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report
# Cargar dataset
dataset_path = 'processed_dataset.csv'
data = pd.read_csv(dataset_path)
# Preprocesamiento
# Verificar si hay valores faltantes y eliminarlos
data = data.dropna()


# Codificación de variables categóricas
X = pd.get_dummies(data.drop('class', axis=1), drop_first=True)
y = data['class']

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42,stratify=y)

# Modelos
# 1. Regresión Logística
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, log_preds)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, log_preds))
