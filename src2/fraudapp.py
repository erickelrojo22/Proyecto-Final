import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title='FraudApp', layout='centered')

# Cargar datos (aquí asumimos que ya tienes el DataFrame `df`)
@st.cache_data
def load_data():
    # Simulación de carga de datos
    data = pd.read_csv('../data/processed/df.csv')
    return data

df = load_data()

# Cargar modelo preentrenado
model = joblib.load('../models/randforest_classifier_.sav')

# Variables
num_variables = ["Transaction_Amount", "Transaction_Type_n", "Account_Balance", "Device_Type_n",
                 "Merchant_Category_n", "Previous_Fraudulent_Activity", "Avg_Transaction_Amount_7d",
                 "Failed_Transaction_Count_7d", "Card_Type_n", "Transaction_Distance", 
                 "Authentication_Method_n", "Risk_Score"]

variable_names_es = {
    "Transaction_Amount": "Monto de la Transacción (USD)",
    "Transaction_Type_n": "Tipo de Transacción",
    "Account_Balance": "Balance en la Cuenta (USD)",
    "Device_Type_n": "Tipo de Dispositivo",
    "Merchant_Category_n": "Categoría del Comerciante",
    "Previous_Fraudulent_Activity": "Actividad Fraudulenta Previa",
    "Avg_Transaction_Amount_7d": "Promedio de Monto de Transacciones (7 días) (USD)",
    "Failed_Transaction_Count_7d": "Cantidad de Transacciones Fallidas (7 días)",
    "Card_Type_n": "Tipo de Tarjeta",
    "Transaction_Distance": "Distancia de la Transacción (millas)",
    "Authentication_Method_n": "Método de Autenticación",
    "Risk_Score": "Puntaje de Riesgo"
}

# Opciones para las variables categóricas
categorical_options = {
    "Transaction_Type_n": ["POS", "Bank Transfer", "ATM Withdrawal", "Online"],
    "Device_Type_n": ["Mobile", "Tablet", "Laptop"],
    "Merchant_Category_n": ["Restaurants", "Electronics", "Clothing", "Travel", "Groceries"],
    "Card_Type_n": ["Amex", "Mastercard", "Visa", "Discover"],
    "Authentication_Method_n": ["OTP", "PIN", "Biometric", "Password"],
    "Risk_Score": [0, 1],
    "Failed_Transaction_Count_7d": [1, 2, 3, 4]
}

# Mapas de codificación para convertir texto a valores numéricos
encoding_maps = {
    "Transaction_Type_n": {"POS": 0, "Bank Transfer": 1, "ATM Withdrawal": 2, "Online": 2},
    "Device_Type_n": {"Mobile": 0, "Tablet": 1, "Laptop": 2},
    "Merchant_Category_n": {"Restaurants": 0, "Electronics": 1, "Clothing": 2, "Travel": 3, "Groceries": 4},
    "Card_Type_n": {"Amex": 0, "Mastercard": 1, "Visa": 2, "Discover": 3},
    "Authentication_Method_n": {"OTP": 0, "PIN": 1, "Biometric": 2, "Password": 3}
}

# Estilos personalizados
st.image("/Users/erickvanscoit/Downloads/image.png", width=200, caption="Detección de Fraude con IA 🚀")
st.title('🌟 Detección de Fraude con Random Forest 🌟')
st.markdown("""
    <style>
    /* Forzar el fondo de la aplicación */
    .stApp {
        background-color: #003049; /* Azul muy oscuro */
        color: #0096c7; /* Azul vibrante para el texto */
    }

    /* Fondo del cuerpo (body) para navegadores específicos */
    body {
        background-color: #03045e !important; /* Azul muy oscuro */
        color: #0096c7; /* Azul vibrante para el texto */
    }

    /* Botones (stButton) */
    .stButton > button {
        background-color: #0077b6; /* Azul medio */
        color: white; /* Texto en blanco */
        border-radius: 5px; /* Bordes redondeados */
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #023e8a; /* Azul intermedio al pasar el cursor */
        color: white;
    }

    /* Selectbox (stSelectbox) */
    .stSelectbox label {
        color: #0096c7; /* Azul vibrante para las etiquetas */
        font-weight: bold;
    }
    .stSelectbox div[data-baseweb="select"] {
        border: 2px solid #0077b6 !important; /* Borde azul medio */
        border-radius: 5px; /* Bordes redondeados */
    }

    /* TextInput (stTextInput) */
    .stTextInput label {
        color: #023e8a; /* Azul intermedio para las etiquetas */
        font-weight: bold;
    }

    /* Slider (stSlider) */
    .stSlider .st-c7 {
        color: #0096c7; /* Azul vibrante para el valor del slider */
    }
    .stSlider .st-bp {
        background: #0077b6; /* Color de la barra activa */
    }
    .stSlider .st-cz {
        background: #023e8a; /* Color de la barra inactiva */
    }
    </style>
""", unsafe_allow_html=True)




# Título de la aplicación
st.title('FraudApp')

# Subtítulo de la predicción
st.subheader('Hacer una predicción')

# Entrada de datos del usuario
input_data = []
for feature in num_variables:
    if feature in categorical_options:
        value = st.selectbox(f'{variable_names_es[feature]}', categorical_options[feature])
        if feature in encoding_maps:
            value = encoding_maps[feature].get(value, value)  # Convertir texto a valor numérico si aplica
    elif feature == "Transaction_Distance":
        value = st.slider(f'{variable_names_es[feature]}', 0, 5000, 0)
    else:
        value = st.text_input(f'{variable_names_es[feature]}', value="0")
        try:
            value = float(value)
        except ValueError:
            st.warning(f'Por favor, introduce un valor numérico válido para {variable_names_es[feature]}')
            value = 0.0
    input_data.append(value)

# Botón de predicción
if st.button('Predecir'):
    prediction = model.predict([input_data])
    st.write('Fraude detectado' if prediction[0] == 1 else 'Transacción legítima')

