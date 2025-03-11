import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar datos (aquí asumimos que ya tienes el DataFrame `df`)
@st.cache_data
def load_data():
    # Simulación de carga de datos
    data = pd.read_csv('../data/processed/df.csv')
    return data

df = load_data()

# Variables
num_variables = ["Transaction_Amount", "Transaction_Type_n", "Account_Balance", "Device_Type_n", "Merchant_Category_n", "Previous_Fraudulent_Activity", "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d", "Card_Type_n", "Transaction_Distance", "Authentication_Method_n", "Risk_Score"]

variable_names_es = {
    "Transaction_Amount": "Monto de la Transacción",
    "Transaction_Type_n": "Tipo de Transacción",
    "Account_Balance": "Balance en la Cuenta",
    "Device_Type_n": "Tipo de Dispositivo",
    "Merchant_Category_n": "Categoría del Comerciante",
    "Previous_Fraudulent_Activity": "Actividad Fraudulenta Previa",
    "Avg_Transaction_Amount_7d": "Promedio de Monto de Transacciones (7 días)",
    "Failed_Transaction_Count_7d": "Cantidad de Transacciones Fallidas (7 días)",
    "Card_Type_n": "Tipo de Tarjeta",
    "Transaction_Distance": "Distancia de la Transacción",
    "Authentication_Method_n": "Método de Autenticación",
    "Risk_Score": "Puntaje de Riesgo"
}

# Opciones para las variables categóricas
categorical_options = {
    "Transaction_Type_n": ["POS", "Bank Transfer", "ATM Withdrawal", "Online"],
    "Device_Type_n": ["Mobile", "Tablet", "Laptop"],
    "Merchant_Category_n": ["Restaurants", "Electronics", "Clothing", "Travel", "Groceries"],
    "Card_Type_n": ["Amex", "Mastercard", "Visa", "Discover"],
    "Authentication_Method_n": ["OTP", "PIN", "Biometric","Password"]
}

# Mapas de codificación para convertir texto a valores numéricos
encoding_maps = {
    "Transaction_Type_n": {"POS": 0, "Bank Transfer": 1, "ATM Withdrawal": 2, "Online": 2},
    "Device_Type_n": {"Mobile": 0, "Tablet": 1, "Laptop": 2},
    "Merchant_Category_n": {"Restaurants": 0, "Electronics": 1, "Clothing": 2, "Travel": 3, "Groceries": 4},
    "Card_Type_n": {"Amex": 0, "Mastercard": 1, "Visa": 2, "Discover": 3},
    "Authentication_Method_n": {"OTP": 0, "PIN": 1, "Biometric": 2, "Password": 3}
}

X = df[num_variables]
y = df["Fraud_Label"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Interfaz de usuario con Streamlit
st.title('Detección de Fraude con Random Forest')

if st.button('Evaluar modelo'):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Precisión del modelo: {accuracy:.2%}')

    st.subheader('Matriz de confusión')
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader('Reporte de clasificación')
    st.text(classification_report(y_test, y_pred))

st.subheader('Hacer una predicción')
input_data = []
for feature in num_variables:
    if feature in categorical_options:
        value = st.selectbox(f'{variable_names_es[feature]}', categorical_options[feature])
        value = encoding_maps[feature][value]  # Convertir texto a valor numérico
    else:
        value = st.text_input(f'{variable_names_es[feature]}', value="0")
        try:
            value = float(value)
        except ValueError:
            st.warning(f'Por favor, introduce un valor numérico válido para {variable_names_es[feature]}')
            value = 0.0
    input_data.append(value)

if st.button('Predecir'):
    prediction = model.predict([input_data])
    st.write('Fraude detectado' if prediction[0] == 1 else 'Transacción legítima')
