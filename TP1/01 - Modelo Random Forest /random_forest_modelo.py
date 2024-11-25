
# Random Forest - Modelo de Clasificación
import pandas as pd # Manejo y análisis de datos estructurados.
from sklearn.model_selection import train_test_split # Divide los datos en conjuntos de entrenamiento y prueba.
from sklearn.ensemble import RandomForestClassifier # Implementación del algoritmo Random Forest para clasificación.
from sklearn.metrics import classification_report, accuracy_score # Métricas clave para evaluar el desempeño del modelo.

# 1. Cargar el conjunto de datos
# Reemplaza 'processed_dataset.csv' con la ruta a tu archivo de datos
dataset_path = 'processed_dataset.csv'
df = pd.read_csv(dataset_path)

# 2. Separar características (X) y la variable objetivo (y)
X = df.drop(columns=['class'])  # Características
y = df['class']  # Variable objetivo

# 3. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Entrenar el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Realizar predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# 6. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 7. Imprimir resultados
print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación:")
print(report)
