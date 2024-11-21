import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold

# Cargar el dataset
df = pd.read_csv('Yellow_Submarine.csv')

# Verificar si el archivo se carga correctamente
print("Columnas del DataFrame:")
print(df.columns)

# Imprimir el número de columnas del DataFrame
print("Número de columnas del DataFrame:", df.shape[1])

# Ordenar las columnas del DataFrame alfabéticamente
df = df.reindex(sorted(df.columns), axis=1)

# Identificar columnas nominales y ordinales
nominal_cols = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-attachment', 'gill-spacing', 'gill-color', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-type', 'spore-print-color', 'population', 'habitat']
ordinal_cols = ['bruises', 'gill-size', 'stalk-shape', 'ring-number']

# Definir el preprocesamiento para columnas nominales y ordinales
preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', OneHotEncoder(), nominal_cols),
        ('ordinal', OrdinalEncoder(), ordinal_cols)
    ])

# Verificar si la columna 'class' está presente
if 'class' not in df.columns:
    raise KeyError("La columna 'class' no se encuentra en el DataFrame.")

# Separar las características y la variable objetivo
X = df.drop('class', axis=1)
y = df['class']

# Crear un pipeline que incluya el preprocesamiento, la eliminación de baja varianza y el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('variance_threshold', VarianceThreshold(threshold=0.1)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Análisis de características
# Extraer el modelo entrenado del pipeline
model = pipeline.named_steps['classifier']
# Obtener las importancias de las características
feature_importances = model.feature_importances_
# Obtener los nombres de las características después del preprocesamiento y eliminación de baja varianza
preprocessed_features = pipeline.named_steps['preprocessor'].transform(X_train)
selected_features = pipeline.named_steps['variance_threshold'].get_support(indices=True)
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
selected_feature_names = feature_names[selected_features]
# Crear un DataFrame para mostrar las importancias de las características
feature_importances_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances})
# Ordenar el DataFrame por importancia descendente
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
print("Feature importances:")
print(feature_importances_df)

# Eliminar características con baja importancia
threshold = 0.005
features_to_keep = feature_importances_df[feature_importances_df['Importance'] >= threshold]['Feature']
print("Características a mantener:", features_to_keep)

# Actualizar el preprocesador para mantener solo las características importantes
preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', OneHotEncoder(), [col for col in nominal_cols if f'nominal__{col}' in features_to_keep.values]),
        ('ordinal', OrdinalEncoder(), [col for col in ordinal_cols if f'ordinal__{col}' in features_to_keep.values])
    ])

# Crear un nuevo pipeline con las características seleccionadas
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entrenar el modelo con las características seleccionadas
pipeline.fit(X_train, y_train)

# Evaluar el modelo con las características seleccionadas
y_pred = pipeline.predict(X_test)
print("Accuracy después de eliminar características:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Procesar el dataset completo utilizando el preprocesador del pipeline
processed_data = pipeline.named_steps['preprocessor'].fit_transform(X)

# Convertir el dataset procesado a un DataFrame
processed_df = pd.DataFrame(processed_data, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())

# Agregar la columna objetivo 'class' al DataFrame procesado
processed_df['class'] = y.values

# Guardar el DataFrame procesado en un archivo CSV
processed_df.to_csv('Processed_Yellow_Submarine.csv', index=False)

print("El dataset procesado ha sido guardado en 'Processed_Yellow_Submarine.csv'.")
