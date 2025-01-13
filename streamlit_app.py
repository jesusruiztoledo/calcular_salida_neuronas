import numpy as np
import streamlit as st
import funciones as f
from neuron import Neuron

st.title("Simulador de Neurona")
st.image("neurona.jpg")

# Selección del número de entradas/pesos
num_inputs = st.slider("Elige el número de entradas/pesos que tendrá la neurona", min_value=1, max_value=10, value=3)

# Entrada de pesos en columnas
st.subheader("Pesos")
weights = []
cols = st.columns(num_inputs)
for i in range(num_inputs):
    with cols[i]:
        weight = st.number_input(f"w{i}", value=0.0, step=0.01, format="%.2f")
        weights.append(weight)
        st.write(f"w = {weight}")

# Entrada de datos en columnas
st.subheader("Entradas")
input_data = []
cols = st.columns(num_inputs)
for i in range(num_inputs):
    with cols[i]:
        input_value = st.number_input(f"x{i}", value=0.0, step=0.01, format="%.2f")
        input_data.append(input_value)
        st.write(f"x = {input_value}")



cols = st.columns(2)
with cols[0]:

    # Selección de la función de activación
    st.subheader("Función de activación")
    activation_function = st.selectbox("Elige la función de activación", ["Sigmoid", "Relu", "Tanh", "Binary"])

with cols[1]:
    # Entrada del bias
    st.subheader("Sesgo")
    bias = st.number_input("Introduce el valor del sesgo", value=0.0, step=0.01, format="%.2f")

# Calcular salida
if st.button("Calcular la salida"):
    neuron = Neuron(weights=weights, bias=bias, func=activation_function)
    output = neuron.run(input_data=input_data)
    st.write(f"La salida de la neurona es: {output}")
