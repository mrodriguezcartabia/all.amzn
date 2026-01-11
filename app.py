import streamlit as st
import requests
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime
from scipy.special import comb
from scipy.optimize import minimize_scalar

# Debe estar al comienzo

# --- LÓGICA DE IDIOMA ---
params = st.query_params
idioma = params.get("lang", "en") # Por defecto inglés

texts = {
    "en": {
        "title": "Gold Call Valuator",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ This value corresponds to the Black-Scholes model",
        "sigma_lbl": "Sigma (Volatility)",
        "sigma_cap": "ℹ️ Conservative value based on past data",
        "alpha_lbl": "Alpha",
        "fuente_precio": "ℹ️ Data from API Alpha Vantage",
        "tasa_lbl": "Risk-Free Rate",
        "fuente_tasa": "ℹ️ Source: FRED",
        "venc_msg": "Expires in {} days ({})",
        "val_act": "Current Price",
        "strike_atm": "Strike (At-the-money)",
        "paso_temp": "Time Step",
        "reset": "Reset",
        "recalc": "RECALCULATE",
        "msg_loading": "Running binomial model...",
        "msg_success": "Calculation complete!",
        "graph_title": "Call Price (C) vs Strike (K)",
        "graph_y": "Call Price",
        "info_init": "Click RECALCULATE to generate the visualization.",
        "lbl_ingresar": "Enter market data",
        "lbl_guardar": "Save",
        "lbl_hallar": "Find sigma",
        "lbl_res": "Sigma found",
        "lbl_mkt_info": "Enter market prices for each Strike:",
        "precio_mercado": "Price market",
        "msg_error_api": "No connection to API Alpha Vantage",
        "msg_manual_price": "Please enter the price manually to continue.",
        "error_fred": "No connection to FRED",
    },
    "es": {
        "title": "Valuador de Call de Oro",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ Este valor corresponde al modelo de Black-Scholes",
        "sigma_lbl": "Sigma (Volatilidad)",
        "sigma_cap": "ℹ️ Valor conservador basado en datos pasados",
        "alpha_lbl": "Alfa",
        "fuente_precio": "ℹ️ Datos de API Alpha Vantage",
        "tasa_lbl": "Tasa Libre de Riesgo",
        "fuente_tasa": "ℹ️ Fuente: FRED",
        "venc_msg": "Vencimiento en {} días ({})",
        "val_act": "Valor Actual",
        "strike_atm": "Strike At-the-money",
        "paso_temp": "Paso Temporal",
        "reset": "Reestablecer",
        "recalc": "RECALCULAR",
        "msg_loading": "Ejecutando modelo binomial...",
        "msg_success": "¡Cálculo finalizado!",
        "graph_title": "Gráfico de Precio de Call (C) vs Strike (K)",
        "graph_y": "Precio de la opción",
        "info_init": "Presiona RECALCULAR para generar la visualización.",
        "lbl_ingresar": "Ingresar datos de mercado",
        "lbl_guardar": "Guardar",
        "lbl_hallar": "Hallar sigma",
        "lbl_res": "Sigma hallado",
        "lbl_mkt_info": "Introduce los precios de mercado para cada Strike:",
        "precio_mercado": "Valor de mercado",
        "msg_error_api": "Sin conexión con API Alpha Vantage",
        "msg_manual_price": "Por favor, coloque el precio manualmente para continuar.",
        "error_fred": "Sin conexión con FRED",
    },
    "pt": {
        "title": "Valiador de Call de Ouro",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ Este valor corresponde ao modelo Black-Scholes",
        "sigma_lbl": "Sigma (Volatilidade)",
        "sigma_cap": "ℹ️ Valor conservador baseado em dados passados",
        "alpha_lbl": "Alfa",
        "fuente_precio": "ℹ️ Dados da API Alpha Vantage",
        "tasa_lbl": "Taxa Livre de Risco",
        "fuente_tasa": "ℹ️ Fonte: FRED",
        "venc_msg": "Expira em {} dias ({})",
        "val_act": "Preço Atual",
        "strike_atm": "Strike At-the-money",
        "paso_temp": "Passo Temporal",
        "reset": "Restablecer",
        "recalc": "RECALCULAR",
        "msg_loading": "Executando modelo binomial...",
        "msg_success": "Cálculo concluído!",
        "graph_title": "Gráfico de Preço da Call (C) vs Strike (K)",
        "graph_y": "Preço da opção",
        "info_init": "Clique em RECALCULAR para gerar a visualização.",
        "lbl_ingresar": "Insira os dados de mercado",
        "lbl_guardar": "Salvar",
        "lbl_hallar": "Encontre sigma",
        "lbl_res": "Sigma encontrado",
        "lbl_mkt_info": "Insira os preços de mercado para cada Strike:",
        "precio_mercado": "Mercado de preços",
        "msg_error_api": "Sem conexão com a API Alpha Vantage",
        "msg_manual_price": "Por favor, insira o preço manualmente para continuar.",
        "error_fred": "Sem conexão com a FRED",
    }
}

t = texts.get(idioma, texts["en"])

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title=t["title"], layout="wide")

# Funciones
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Por si el archivo aún no se sube o falla la lectura

local_css("style.css")

def get_market_data_alpha():
    cache_file = "spot_price.txt"
    # Leamos el archivo
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 7200:
            try:
                with open(cache_file, "r") as f:
                    cached_file = float(f.read())
                return cached_file
            except:
                pass
    # Buscamos en la web
    try:
        api_key = st.secrets["ALPHAVANTAGE_API_KEY"]  
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&apikey={api_key}", headers=headers, timeout=10)
        data = response.json()
        if "Time Series (Daily)" in data:
            # Obtenemos la fecha de cierre más reciente
            ultima_fecha = list(data["Time Series (Daily)"].keys())[0]
            precio_amzn = float(data["Time Series (Daily)"][ultima_fecha]["4. close"])
            
            with open(cache_file, "w") as f:
                f.write(str(precio_amzn))
            return precio_amzn
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None #ojoooooooooooooooooo

def obtener_volatilidad():


def hallar_sigma_optimo(precios_mercado, strikes, S, r, T, beta, paso, param_a):
    def error_cuadratico(sigma_test):
        if sigma_test <= 0: return 1e10
        err = 0
        for i, k in enumerate(strikes):
            # Calculamos el precio del modelo para cada strike con el sigma de prueba
            c_mod = calcular_call(S, k, r, T, sigma_test, beta, paso, param_a)
            err += (c_mod - precios_mercado[i])**2
        return err
    
    # Optimizamos una sola variable (sigma) en un rango de 1% a 200%
    res = minimize_scalar(error_cuadratico, bounds=(0.01, 2.0), method='bounded')
    return res.x 

# --- MOTOR DE CÁLCULO ---
@st.cache_data
def calcular_call(S, K, r, T, sigma, beta, paso, param_a):
    m = int(round(T / paso))
    if m <= 0: m = 1
    dt = T / m
    u = np.exp(param_a * sigma * (paso**beta))
    d = u**(-1/param_a**2)
    tasa = np.exp(r * dt)
    p = (tasa - d) / (u - d)
    p = max(min(p, 1.0), 0.0)
    suma_binomial = 0
    for k in range(m + 1):                     
        prob = comb(m, k) * (p**k) * ((1-p)**(m-k))
        st_k = S * (u**k) * (d**(m-k))
        payoff = max(st_k - K, 0)
        suma_binomial += prob * payoff
    return np.exp(-r * T) * suma_binomial

# --- ESTADO DE SESIÓN ---

if 'tiempo_total' not in st.session_state:
  st.session_state.tiempo_total = 1
if 'valor_amzn' not in st.session_state:
  st.session_state.valor_amzn = get_market_data_alpha()
if 'paso_val' not in st.session_state:
  st.session_state.paso_val = 0.1
if 'market_cache' not in st.session_state:
  st.session_state.market_cache = None
if 'tasa_cache' not in st.session_state:
  st.session_state.tasa_cache = get_fred_risk_free_rate() #cuidado
if 'data_grafico' not in st.session_state:
  st.session_state.data_grafico = None
if 'mostrar_editor' not in st.session_state:
  st.session_state.mostrar_editor = False
if 'sigma_hallado' not in st.session_state:
  st.session_state.sigma_hallado = None
if 'precios_mercado' not in st.session_state:
  st.session_state.precios_mercado = [0.0] * 7

# --- INTERFAZ ---




