import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
# Configuración de página
st.set_page_config(page_title='FraudApp',
                   page_icon='../data/processed/fraudapp.png',
                   layout='centered')
# Cargar modelo
# Ruta base desde la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta al modelo
MODEL_PATH = BASE_DIR / 'models/randforest_classifier_.sav'

# Cargar el modelo
model = joblib.load(MODEL_PATH)

#model = joblib.load('../models/randforest_classifier_.sav')
# Variables en el orden correcto para el modelo
FEATURE_ORDER = [
    "Transaction_Amount", "Transaction_Type_n", "Account_Balance",
    "Device_Type_n", "Merchant_Category_n", "Previous_Fraudulent_Activity",
    "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
    "Card_Type_n", "Transaction_Distance", "Authentication_Method_n", "Risk_Score"
]
# Traducciones al español
VARIABLE_TRANSLATIONS = {
    "Transaction_Amount": "Monto de la Transacción (USD)",
    "Transaction_Type_n": "Tipo de Transacción",
    "Account_Balance": "Balance en la Cuenta (USD)",
    "Device_Type_n": "Tipo de Dispositivo",
    "Merchant_Category_n": "Categoría del Comerciante",
    "Previous_Fraudulent_Activity": "Actividad Fraudulenta Previa",
    "Avg_Transaction_Amount_7d": "Promedio de Monto de Transacciones (7 días) (USD)",
    "Failed_Transaction_Count_7d": "Cantidad de Transacciones Fallidas (7 días)",
    "Card_Type_n": "Tipo de Tarjeta",
    "Transaction_Distance": "Distancia de la Transacción (Millas)",
    "Authentication_Method_n": "Método de Autenticación",
    "Risk_Score": "Puntaje de Riesgo"
}
# Mapeos actualizados
ENCODING_MAPS = {
    "Transaction_Type_n": {
        "POS": 0,
        "Bank Transfer": 1,
        "ATM Withdrawal": 2,
        "Online": 3
    },
    "Device_Type_n": {
        "Mobile": 0,
        "Tablet": 1,
        "Laptop": 2
    },
    "Merchant_Category_n": {
        "Restaurants": 0,
        "Electronics": 1,
        "Clothing": 2,
        "Travel": 3,
        "Groceries": 4
    },
    "Card_Type_n": {
        "Amex": 0,
        "Mastercard": 1,
        "Visa": 2,
        "Discover": 3
    },
    "Authentication_Method_n": {
        "OTP": 0,
        "PIN": 1,
        "Biometric": 2,
        "Password": 3
    },
    "Previous_Fraudulent_Activity": {"No": 0, "Sí": 1},
    "Risk_Score": {"Bajo": 0, "Alto": 1},
    "Failed_Transaction_Count_7d": {
        "1 Transacción": 1,
        "2 Transacciones": 2,
        "3 Transacciones": 3,
        "4 Transacciones": 4
    }
}
# Estilos CSS con animación
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFFFFF, #003049, #0096C7);
        background-size: 400% 400%;
        animation: gradientAnimation 10s ease infinite;
        color: #FFFFFF;
    }
    @keyframes gradientAnimation {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp::before {
        content: '';
        position: fixed;
        width: 100%;
        height: 100%;
        background-image: url('../data/processed/logo.webp');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.2;
        z-index: -1;
    }
    </style>
""", unsafe_allow_html=True)
# Interfaz de usuario
with st.container():
    # Encabezado centrado
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image('../data/processed/fraudapp.png', width=200)
    st.markdown("<h2>Haz tu predicción Bancaria</h2>", unsafe_allow_html=True)
    st.markdown("<h3>Ingrese su información</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    input_data = []
    for feature in FEATURE_ORDER:
        label = VARIABLE_TRANSLATIONS[feature]
        # Manejo de campos especiales
        if feature in ENCODING_MAPS:
            options = list(ENCODING_MAPS[feature].keys())
            selected = st.selectbox(label, options)
            encoded_value = ENCODING_MAPS[feature][selected]
            input_data.append(encoded_value)
        elif feature == "Transaction_Distance":
            value = st.slider(label, 0, 5000, 0)
            input_data.append(value)
        else:
            # Campos numéricos estándar
            value = st.number_input(label, value=0.0, step=0.01)
            input_data.append(float(value))
    # Botón de predicción
    if st.button('Predecir', help="Haz clic para realizar la predicción"):
        try:
            prediction = model.predict([input_data])
            probability = model.predict_proba([input_data])[0][1]
            # Mostrar resultados
            if prediction[0] == 1:
                st.error(f':luz_giratoria: Posible fraude detectado (probabilidad: {probability:.2%})')
            else:
                st.success(f':marca_de_verificación_blanca: Transacción legítima (probabilidad: {1 - probability:.2%})')
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")

