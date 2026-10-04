import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 1. Cargar datos de ejemplo
# Crear un conjunto de datos simulado de productos
data = pd.DataFrame({
    'Feature1': [1.0, 2.0, 1.5, 8.0, 9.0, 8.5],
    'Feature2': [1.0, 1.8, 1.2, 8.0, 8.8, 9.0],
    'Feature3': [0.5, 0.8, 0.6, 7.5, 8.0, 8.2],
    'label': ['Producto_A', 'Producto_A', 'Producto_A', 'Producto_B', 'Producto_B', 'Producto_B']
})

features = data[['Feature1', 'Feature2', 'Feature3']]
labels = data['label']

# 2. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 3. Crear y entrenar el modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Realizar predicciones
y_pred = model.predict(X_test)

# 5. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 6. Función de recomendación
def recommend(product_features):
    prediction = model.predict(product_features)
    return prediction

# Ejemplo de uso de la función de recomendación
example_product = [[1.2, 1.9, 0.7]]
recommended_product = recommend(example_product)
print(f"Recommended Product: {recommended_product[0]}")
