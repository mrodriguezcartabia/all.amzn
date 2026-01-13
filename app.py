import streamlit as st
import requests
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import os
import time
import json
import plotly.graph_objects as go
from datetime import datetime
from scipy.special import comb
from scipy.optimize import minimize_scalar

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
        "fuente_precio": "Data from API Alpha Vantage",
        "tasa_lbl": "Rate",
        "fuente_tasa": "Data from FRED",
        "venc_msg": "Expires in {} days ({})",
        "val_act": "Current Price",
        "strike_atm": "Strike (At-the-money)",
        "paso_temp": "Time Step",
        "reset": "Reset",
        "recalc": "RECALCULATE",
        "msg_loading": "Running Binomial model",
        "msg_loading_2": "Running Least squares",
        "msg_success": "Calculation complete!",
        "graph_title": "Call Price (C) vs Strike (K)",
        "graph_y": "Price",
        "info_init": "Click RECALCULATE to generate the visualization.",
        "lbl_ingresar": "Enter market data",
        "lbl_guardar": "Save",
        "lbl_hallar": "FIND VARIABLE",
        "lbl_res": "Sigma found",
        "lbl_mkt_info": "Enter market prices for each Strike:",
        "precio_mercado": "Price market",
        "msg_error_api": "No connection to API Alpha Vantage",
        "msg_manual_price": "Please enter the price manually to continue.",
        "error_fred": "No connection to FRED",
        "error_fred": "No connection to FRED",
        "parar_precio": "Error connecting to Alpha Vantage. Please enter the price manually:",
        "parar_tasa": "Error connecting to FRED. Please enter the rate manually:",
        "parar_sigma": "Error connecting to Alpha Vantage. Please enter the volatility manually:",
        "dias": "Days",
        "seleccionar": "Select",
    },
    "es": {
        "title": "Valuador de Call de Oro",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ Este valor corresponde al modelo de Black-Scholes",
        "sigma_lbl": "Sigma (Volatilidad)",
        "sigma_cap": "ℹ️ Valor conservador basado en datos pasados",
        "alpha_lbl": "Alfa",
        "fuente_precio": "Datos de API Alpha Vantage",
        "tasa_lbl": "Tasa",
        "fuente_tasa": "Datos de FRED",
        "venc_msg": "Vencimiento en {} días ({})",
        "val_act": "Valor Actual",
        "strike_atm": "Strike At-the-money",
        "paso_temp": "Paso Temporal",
        "reset": "Reestablecer",
        "recalc": "RECALCULAR",
        "msg_loading": "Ejecutando el modelo binomial",
        "msg_loading_2": "Ejecutando mínimos cuadrados",
        "msg_success": "¡Cálculo finalizado!",
        "graph_title": "Gráfico de Precio de Call (C) vs Strike (K)",
        "graph_y": "Precio",
        "info_init": "Presiona RECALCULAR para generar la visualización.",
        "lbl_ingresar": "Ingresar datos de mercado",
        "lbl_guardar": "Guardar",
        "lbl_hallar": "HALLAR VARIABLE",
        "lbl_res": "Sigma hallado",
        "lbl_mkt_info": "Introduce los precios de mercado para cada Strike:",
        "precio_mercado": "Valor de mercado",
        "msg_error_api": "Sin conexión con API Alpha Vantage",
        "msg_manual_price": "Por favor, coloque el precio manualmente para continuar.",
        "error_fred": "Sin conexión con FRED",
        "parar_precio": "Error al conectar con Alpha vetage. Por favor, introduzca el precio manualmente:",
        "parar_tasa": "Error al conectar con FRED. Por favor, introduzca la tasa manualmente:",
        "parar_sigma": "Error al conectar con Alpha vetage. Por favor, introduzca la volatilidad anual manualmente:",
        "dias": "Días",
        "seleccionar": "Selecccionar",
    },
    "pt": {
        "title": "Valiador de Call de Ouro",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ Este valor corresponde ao modelo Black-Scholes",
        "sigma_lbl": "Sigma (Volatilidade)",
        "sigma_cap": "ℹ️ Valor conservador baseado em dados passados",
        "alpha_lbl": "Alfa",
        "fuente_precio": "Dados da API Alpha Vantage",
        "tasa_lbl": "Taxa",
        "fuente_tasa": "Dados da FRED",
        "venc_msg": "Expira em {} dias ({})",
        "val_act": "Preço Atual",
        "strike_atm": "Strike At-the-money",
        "paso_temp": "Passo Temporal",
        "reset": "Restablecer",
        "recalc": "RECALCULAR",
        "msg_loading": "Executando modelo binomial",
        "msg_loading_2": "Executando método dos mínimos quadrados",
        "msg_success": "Cálculo concluído!",
        "graph_title": "Gráfico de Preço da Call (C) vs Strike (K)",
        "graph_y": "Preço",
        "info_init": "Clique em RECALCULAR para gerar a visualização.",
        "lbl_ingresar": "Insira os dados de mercado",
        "lbl_guardar": "Salvar",
        "lbl_hallar": "ENCONTRE VARIÁVEL",
        "lbl_res": "Sigma encontrado",
        "lbl_mkt_info": "Insira os preços de mercado para cada Strike:",
        "precio_mercado": "Mercado de preços",
        "msg_error_api": "Sem conexão com a API Alpha Vantage",
        "msg_manual_price": "Por favor, insira o preço manualmente para continuar.",
        "error_fred": "Sem conexão com a FRED",
        "error_fred": "Sem conexão com a FRED",
        "parar_precio": "Erro ao conectar com Alpha Vantage. Por favor, insira o preço manualmente:",
        "parar_tasa": "Erro ao conectar com a FRED. Por favor, insira a taxa manualmente:",
        "parar_sigma": "Erro ao conectar com a Alpha Vantage. Por favor, insira a volatilidade anual manualmente:",
        "dias": "Dias",
        "seleccionar": "Selecionar",
    }
}

t = texts.get(idioma, texts["en"])

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title=t["title"], layout="wide")

# FUNCIONES
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
        response = requests.get(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AMZN&apikey={api_key}", headers=headers, timeout=10)
        data = response.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            precio_amzn = float(data["Global Quote"]["05. price"])
            
            with open(cache_file, "w") as f:
                f.write(str(precio_amzn))
            return precio_amzn
    except:
        pass
    if st.session_state.valor_temporal is None:
        parar_juego(t["parar_precio"])
    precio = st.session_state.valor_temporal
    st.session_state.valor_temporal = None
    return precio

def get_fred_risk_free_rate():
    cache_file = "risk_free.txt"

    # Leamos el archivo
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 10800:
            try:
                with open(cache_file, "r") as f:
                    cached_file = float(f.read())
                return cached_file
            except:
                pass
    # Buscamos en internet
    try:
        api_key = st.secrets["FRED_API_KEY"]
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=DTB4WK&api_key={api_key}&file_type=json&sort_order=desc&limit=5", headers=headers)
        data = response.json()
        # Buscamos el primer valor útil (por si es feriado)
        for obs in data['observations']:
            val =obs['value']
            if val != ".":
                return float(val) / 100
    except:
        pass
    if st.session_state.valor_temporal is None:
        parar_juego(t["parar_tasa"])
    precio = st.session_state.valor_temporal
    st.session_state.valor_temporal = None
    return precio

def get_volatility_data_alpha():
    cache_file = "volatility_data.txt"
    # Leamos el archivo
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 72000:
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
            # Extraer precios de cierre
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df = df['4. close'].astype(float).sort_index()
            log_returns = np.log(df / df.shift(1)).dropna() # 1. Calcular rendimientos logarítmicos
            sigma_diario = log_returns.std() # 2. Desviación estándar diaria
            sigma_anual = sigma_diario * np.sqrt(252) # 3. Anualizar (252 días de trading)
            with open(cache_file, "w") as f:
                f.write(str(sigma_anual))
            return sigma_anual
    except:
        pass
    if st.session_state.valor_temporal is None:
        parar_juego(t["parar_sigma"])
    sigma_anual = st.session_state.valor_temporal
    st.session_state.valor_temporal = None
    return sigma_anual            

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

def parar_juego(cartel):
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown(f"""
            <div class="overlay-card-static">
                <h2 style="color: #DAA520; text-align: center;">{t['title']}</h2>
                <p style="color: white; text-align: center;">{cartel}</p>
            </div>
        """, unsafe_allow_html=True)
        with st.form(key="manual_input_form", clear_on_submit=True):
            valor_temporal = st.number_input(t["val_act"], value=None, placeholder="")
            submit_button = st.form_submit_button("ENTER", use_container_width=True, type="primary")
            if submit_button:
                if valor_temporal is not None and  valor_temporal > 0:
                    st.session_state.valor_temporal = valor_temporal
                    st.rerun()
                else:
                    st.warning(t["msg_manual_price"])
        st.stop()

# Funciones del cálculo central
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

def optimizar_parametro(target_param, precios_mercado, strikes, S, r, T, sigma, beta, paso, param_a):
    def error_cuadratico(valor_test):
        if valor_test <= 0: return 1e10
        err = 0
        # Asignamos el valor de prueba al parámetro elegido
        p = {
            "sigma": valor_test if target_param == "Sigma" else sigma,
            "beta": valor_test if target_param == "Beta" else beta,
            "alpha": valor_test if target_param == t["alpha_lbl"] else param_a,
            "tasa": valor_test if target_param == t["tasa_lbl"] else r
        }
        
        for i in range(len(strikes)):
            c_mod = calcular_call(S, strikes[i], p["tasa"], T, p["sigma"], p["beta"], paso, p["alpha"])
            err += (c_mod - precios_mercado[i])**2
        return err

    # Definimos rangos lógicos según el parámetro
    bounds = {
        "Sigma": (0.01, 3.0),
        "Beta": (0.01, 10.0),
        t["alpha_lbl"]: (0.1, 5.0),
        t["tasa_lbl"]: (0.0, 2.0)
    }
    
    res = minimize_scalar(error_cuadratico, bounds=bounds[target_param], method='bounded')
    return res.x

# --- ESTADO DE SESIÓN ---
valor_paso_original = 0.1
# Creamos una variable que sirve en caso de que falle la comunicación con Alpha v o FRED
if 'valor_temporal' not in st.session_state:
    st.session_state.valor_temporal = None
if 'tiempo_total' not in st.session_state:
  st.session_state.tiempo_total = 1
if 'precio_AMZN' not in st.session_state:
  st.session_state.precio_AMZN = get_market_data_alpha()
if 'paso_val' not in st.session_state:
  st.session_state.paso_val = valor_paso_original
if 'tasa_cache' not in st.session_state:
  st.session_state.tasa_cache = get_fred_risk_free_rate() 
if 'data_grafico' not in st.session_state:
  st.session_state.data_grafico = None
if 'mostrar_editor' not in st.session_state:
  st.session_state.mostrar_editor = False
if 'sigma_hallado' not in st.session_state:
  st.session_state.sigma_hallado = get_volatility_data_alpha()
# Ahora iniciamos todas las variables que necesitamos para optimizar
if 'variable_optimizada' not in st.session_state:
    st.session_state.variable_optimizada = None
if 'resultado_opt' not in st.session_state:
    st.session_state.resultado_opt = None
if 'sigma_opt' not in st.session_state:
    st.session_state.sigma_opt = 0.0
if 'beta_opt' not in st.session_state:
    st.session_state.beta_opt = 0.0
if 'alpha_opt' not in st.session_state:
    st.session_state.alpha_opt = 0.0
if 'tasa_opt' not in st.session_state:
    st.session_state.tasa_opt = 0.0

# --- INTERFAZ ---
col1, col2, col3 = st.columns(3)
with col1:
    param_a = st.number_input(t["alpha_lbl"], value=1.0, step=0.01, min_value=0.1, max_value=10.0)
    sigma = st.number_input(t["sigma_lbl"], value=float(st.session_state.sigma_hallado), format="%.4f", min_value=0.001, max_value=3.0)
    st.caption(f"{t['fuente_precio']} = {st.session_state.sigma_hallado:.4f}")

with col2:
    beta = st.number_input("Beta", value=0.5, step=0.01, min_value=0.0, max_value=10.0)
    tasa_r = st.number_input(t["tasa_lbl"], value=float(st.session_state.tasa_cache), format="%.4f", min_value=0.0, max_value=10.0)
    st.caption(f"{t['fuente_tasa']} = {st.session_state.tasa_cache:.4f}")

with col3:
    dias = st.number_input(t["dias"], value=1.0, step=1.0, min_value=1.0, max_value=365.0)
    precio_accion = st.number_input(t["val_act"], value=float(st.session_state.precio_AMZN), step=0.01, min_value=1.0)
    st.caption(f"{t['fuente_precio']} = {st.session_state.precio_AMZN}")

# Variables
st.divider()
tiempo_T = dias / 365
# Introducimos los Strikes
strike = round(precio_accion / 2.5) * 2.5
rango_strikes = np.arange(strike - 7.5, strike + 8, 2.5)
if 'precios_mercado' not in st.session_state:
  st.session_state.precios_mercado = [0.0] * len(rango_strikes)

herramientas, grafico = st.columns([1, 2])
with herramientas:
    st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="
                color: #cbd5e0;
                font-size: 0.9rem;
                font-weight: 500;
                margin-bottom: 8px;">
                {t["paso_temp"]}
            </div>
            <div style="
                height: 42px;
                background-color: #1e293b;;
                color: #fafafa;
                padding: 0px 12px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 0.5rem;
                display: flex;
                align-items: center;
                font-size: 1rem;
                font-family: 'Inter', sans-serif;">
                {st.session_state.paso_val:.6f}
            </div>
        </div>
    """, unsafe_allow_html=True)
    

    # Botones de paso temporal
    boton1, boton2 = st.columns([1, 1.5])
    with boton1:
        if st.button("x10⁻¹") and st.session_state.paso_val > 10**(-5):
            st.session_state.paso_val *= 0.1
            st.rerun()
    with boton2:
        if st.button(t["reset"], key="btn-reset"):
            st.session_state.paso_val = valor_paso_original
            st.rerun()

    # Botón para el cálculo      
    btn_recalcular = st.button(t["recalc"], type="primary", use_container_width=True)

    # Creamos el entorno para el ingreso de datos
    with st.popover(t["lbl_ingresar"], use_container_width=True):
        st.write(t["lbl_mkt_info"])
        
        # Creamos un formulario interno
        with st.form("form_mercado"):
            if len(st.session_state.precios_mercado) != len(rango_strikes):
                st.session_state.precios_mercado = [0.0] * len(rango_strikes)                
            df_editor = pd.DataFrame({
                "Strike": rango_strikes, 
                t["precio_mercado"]: st.session_state.precios_mercado
            })
            
            # El editor dentro del formulario no dispara re-ejecuciones automáticas
            edited_df = st.data_editor(
                df_editor, 
                hide_index=True, 
                use_container_width=True,
                num_rows="fixed",
                column_config={"Strike": st.column_config.NumberColumn(disabled=True),
                t["precio_mercado"]: st.column_config.NumberColumn(min_value=0.0)}
            )
            
            # 2. Botón para confirmar los cambios
            submit_save = st.form_submit_button(t["lbl_guardar"], use_container_width=True)
            
            if submit_save:
                # Solo aquí guardamos los datos en el estado global
                st.session_state.precios_mercado = edited_df[t["precio_mercado"]].tolist()
                st.rerun() # Esto refresca el gráfico con los nuevos puntos
    # Ahora optimizamos
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button(t["alpha_lbl"], use_container_width=True):
            if st.session_state.variable_optimizada != t["alpha_lbl"]:
                st.session_state.variable_optimizada = t["alpha_lbl"]
                st.session_state.resultado_opt = None
    with b2:
        if st.button("Beta", use_container_width=True):
            if st.session_state.variable_optimizada != "Beta":
                st.session_state.variable_optimizada = "Beta"
                st.session_state.resultado_opt = None
    with b3:
        if st.button("Sigma", use_container_width=True):
            if st.session_state.variable_optimizada != "Sigma":
                st.session_state.variable_optimizada = "Sigma"
                st.session_state.resultado_opt = None
    with b4:
        if st.button(t["tasa_lbl"], use_container_width=True):
            if st.session_state.variable_optimizada != t["tasa_lbl"]:
                st.session_state.variable_optimizada = t["tasa_lbl"]
                st.session_state.resultado_opt = None
    # Botón para optimizar
    btn_hallar = st.button(t["lbl_hallar"], type="primary", use_container_width=True)
    if btn_hallar and any(p > 0 for p in st.session_state.precios_mercado) and st.session_state.variable_optimizada:
        with st.spinner(t['msg_loading_2']):
            st.session_state.resultado_opt = optimizar_parametro(st.session_state.variable_optimizada, st.session_state.precios_mercado, rango_strikes, precio_accion, tasa_r, 
                                                             tiempo_T, sigma, beta, st.session_state.paso_val, param_a)
        # Otro mensaje temporal de éxito
        if btn_hallar:
            st.toast(t["msg_success"])

    # Resultado del hallado
    variable = st.session_state.variable_optimizada if st.session_state.variable_optimizada else "x"
    valor = f"{st.session_state.resultado_opt:.5f}" if st.session_state.resultado_opt else ""
    st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="
                height: 42px;
                background-color: #1e293b;;
                color: #fafafa;
                padding: 0px 12px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 0.5rem;
                display: flex;
                align-items: center;
                font-size: 1rem;
                font-family: 'Inter', sans-serif;">
                {variable}={valor}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Calculamos los valores del call
if st.session_state.data_grafico is None or btn_recalcular:
    # Indicador de carga activo durante el proceso matemático
    with st.spinner(t['msg_loading']):
        valores_c = []
        for k in rango_strikes:
            c = calcular_call(precio_accion, k, tasa_r, tiempo_T, sigma, beta, st.session_state.paso_val, param_a)
            valores_c.append(c)
        st.session_state.data_grafico = (rango_strikes, valores_c)
    # Mensaje temporal de éxito
    if btn_recalcular:
        st.toast(t["msg_success"])

# Gráfico
with grafico:
    strikes, calls = st.session_state.data_grafico

    # Creamos el gráfico
    fig = go.Figure()
    # Curva del modelo
    fig.add_trace(go.Scatter(
        x=strikes,
        y=calls,
        mode='lines+markers',
        line=dict(color='#FF9900', width=3),
        marker=dict(size=8),
        hovertemplate=f'Strike: %{{x:.2f}}<br>{t["graph_y"]}: %{{y:.2f}}<extra></extra>'
    ))
    # Curva de valores de mercado
    if any(p > 0 for p in st.session_state.precios_mercado):
        fig.add_trace(go.Scatter(
            x=strikes,
            y=st.session_state.precios_mercado,
            mode='lines+markers',
            name=t['precio_mercado'],
            line=dict(color='#000000', width=3),
            marker=dict(size=8),
            hovertemplate=f'Strike: %{{x:.2f}}<br>{t["graph_y"]}: %{{y:.2f}}<extra></extra>'
        ))
    # Estética
    fig.update_layout(
        hovermode='x unified',
        template='plotly_white', # Esto pone el fondo blanco y letras negras
        paper_bgcolor='rgba(0,0,0,0)', # Fondo exterior transparente para adaptarse a Streamlit
        plot_bgcolor='#f0f2f6',          # Fondo interior
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255,255,255,0.5)'
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=rango_strikes,
            title="Strike",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)', # Cuadrícula suave
            linecolor='black'            # Línea del eje negra
        ),
        yaxis=dict(
            title=t["graph_y"],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)', # Cuadrícula suave
            linecolor='black'            # Línea del eje negra
        )
    )

    # Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
