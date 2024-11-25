import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset_path = 'processed_dataset.csv'
df = pd.read_csv(dataset_path)

"""
    Variables para el entrenamiento del modelo:
    - X: 
        Matriz de caracteristicas, contiene los atributos del dataset
        que se utilizarán para predecir la clase de un hongo
    - y: 
        Vector objetivo, contiene la etiqueta asociada a cada muestra 
        en el dataset, representando si un hongo es comestibleo venenoso
"""
X = df.drop(columns=['class'])
y = df['class']

"""
    Separamos el dataset en dos partes, por un lado los datos que van a servir
    para entrenar el modelo, y otros seran usados para realizar pruebas sobre
    el modelo
    - X_train: 
        Matriz de caracteristicas para entrenar el modelo
    - X_test:
        Matriz de caracteristicas, sera utilizar para probar el modelo
    - y_train: 
        Vector objetivo (etiquetas) sera utilizado para entrenar el modelo
    - y_test: 
        Vector objetivo (etiquetas), sera utilizar para probar el modelo
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

"""
    Creamos y entrenamos el modelo, definimos k = 3 como el numero de 
    vecinos
"""
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

"""
    Utilizamos los datos de pruebas separados previamente, (33%) 
    con el objetivo de hacer predicciones sobre el modelo
    previamente entrenado
"""
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))