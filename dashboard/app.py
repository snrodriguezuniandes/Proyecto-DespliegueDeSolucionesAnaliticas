import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta


# CONFIGURACIÓN GENERAL

st.set_page_config(
    page_title="Planeación de Demanda e Inventario",
    page_icon="📦",
    layout="wide"
)

SKUS = {
    '19127': 'Duracell Batteries AA 40pk',
    '294000': 'Storage Box 27gal Professional',
    '455151': 'MS 5-tier Storage Rack'
}

MAPE_MODELOS = {'19127': 24.10, '294000': None, '455151': None}

st.sidebar.title("📦 Mercalia")
pagina = st.sidebar.radio(
    "Navegación",
    ["Estatus general", "Demanda y forecast", "Reabastecimiento"]
)

# PÁGINA 1: ESTATUS GENERAL — ALE

if pagina == "Estatus general":
    st.title("Estatus general")
    st.info("Esta página está siendo desarrollada por Ale.")

# PÁGINA 2: DEMANDA Y FORECAST — GABI

elif pagina == "Demanda y forecast":
    st.title("Demanda y forecast")

    # Selector de SKU y país
    col_sku, col_pais = st.columns([2, 1])
    with col_sku:
        sku_sel = st.selectbox("SKU", options=list(SKUS.keys()),
                               format_func=lambda x: f"{x} — {SKUS[x]}")
    with col_pais:
        df_temp = pd.read_csv(f"DATA/raw/ventashistoricas_{sku_sel}.csv")
        paises = ['Todos los países'] + sorted(df_temp['Country'].dropna().unique().tolist())
        pais_sel = st.selectbox("🌎 País", paises)

    # Cargar datos filtrados
    @st.cache_data
    def cargar_ventas(sku, pais):
        df = pd.read_csv(f"DATA/raw/ventashistoricas_{sku}.csv")
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df = df[df['Quantity'] > 0].dropna(subset=['Quantity'])
        if pais != 'Todos los países':
            df = df[df['Country'] == pais]
        diaria = df.groupby('Transaction_Date')['Quantity'].sum().reset_index()
        diaria.columns = ['fecha', 'demanda']
        return diaria.sort_values('fecha').reset_index(drop=True)

    df_ventas = cargar_ventas(sku_sel, pais_sel)

    # Métricas principales
    mape = MAPE_MODELOS.get(sku_sel)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demanda promedio", f"{df_ventas['demanda'].mean():.0f} u/día")
    col2.metric("Demanda máxima", f"{df_ventas['demanda'].max():.0f} u")
    col3.metric("Días de historial", f"{len(df_ventas):,}")
    col4.metric("MAPE modelo", f"{mape:.1f}%" if mape else "Pendiente")

    st.divider()

    # Rango de fechas para la gráfica histórica
    st.subheader("Demanda histórica")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fecha_ini = st.date_input("Desde", value=df_ventas['fecha'].min().date())
    with col_r2:
        fecha_fin = st.date_input("Hasta", value=df_ventas['fecha'].max().date())

    df_filtrado = df_ventas[
        (df_ventas['fecha'].dt.date >= fecha_ini) &
        (df_ventas['fecha'].dt.date <= fecha_fin)
    ]
    mensual = df_filtrado.set_index('fecha')['demanda'].resample('MS').sum()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mensual.index, mensual.values, linewidth=1.2, color='#1f77b4')
    ax.fill_between(mensual.index, mensual.values, alpha=0.15, color='#1f77b4')
    ax.set_ylabel("Unidades / mes")
    ax.set_title(f"Demanda mensual — SKU {sku_sel} ({pais_sel})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Forecast
    st.subheader("Forecast de demanda")
    if pais_sel != 'Colombia':
        st.info("ℹ️ El forecast está disponible únicamente para Colombia, ya que el modelo fue entrenado con datos de ese país.")

    if sku_sel == '19127':
        try:
            modelo = joblib.load("models/modelo_19127_colombia.pkl")

            # Usar solo Colombia para el forecast (modelo entrenado con Colombia)
            df_colombia = cargar_ventas(sku_sel, 'Colombia')
            historial = df_colombia['demanda'].values.tolist()
            ultima_fecha = df_colombia['fecha'].max()
            fechas_futuras = pd.date_range(
                start=ultima_fecha + timedelta(days=1), periods=30)

            predicciones = []
            hist_temp = historial.copy()
            for fecha in fechas_futuras:
                features = {
                    'dia_semana': fecha.dayofweek,
                    'mes': fecha.month,
                    'semana_anio': fecha.isocalendar()[1],
                    'es_finde': int(fecha.dayofweek >= 5),
                    'lag_7': hist_temp[-7],
                    'lag_14': hist_temp[-14],
                    'lag_30': hist_temp[-30],
                    'media_movil_7': np.mean(hist_temp[-7:]),
                    'media_movil_30': np.mean(hist_temp[-30:]),
                }
                pred = modelo.predict(pd.DataFrame([features]))[0]
                predicciones.append(pred)
                hist_temp.append(pred)

            forecast_df = pd.DataFrame({
                'fecha': fechas_futuras,
                'forecast': [round(p) for p in predicciones]
            })

            # Métricas del forecast
            col1, col2, col3 = st.columns(3)
            col1.metric("Forecast total 30 días",
                        f"{forecast_df['forecast'].sum():,} unidades")
            col2.metric("Forecast diario promedio",
                        f"{forecast_df['forecast'].mean():.0f} unidades")
            col3.metric("Modelo utilizado", "XGBoost")

            # Gráfica histórico + forecast
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ultimos_90 = df_colombia.tail(90)
            ax2.plot(ultimos_90['fecha'], ultimos_90['demanda'],
                     label='Histórico (Colombia)', linewidth=1.2, color='#1f77b4')
            ax2.plot(forecast_df['fecha'], forecast_df['forecast'],
                     label='Forecast XGBoost', linewidth=1.5,
                     color='#ff7f0e', linestyle='--')
            ax2.fill_between(forecast_df['fecha'],
                             forecast_df['forecast'] * 0.85,
                             forecast_df['forecast'] * 1.15,
                             alpha=0.2, color='#ff7f0e',
                             label='Banda ±15%')
            ax2.axvline(ultima_fecha, color='gray',
                        linestyle=':', linewidth=1, label='Hoy')
            ax2.set_ylabel("Unidades / día")
            ax2.set_title(f"Demanda histórica y forecast — SKU {sku_sel} Colombia")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

            # Tabla de forecast
            st.subheader("Forecast por período")
            forecast_df['semana'] = forecast_df['fecha'].dt.to_period('W').astype(str)
            forecast_semanal = forecast_df.groupby('semana')['forecast'].sum().reset_index()
            forecast_semanal.columns = ['Semana', 'Forecast total (unidades)']
            st.dataframe(forecast_semanal, use_container_width=True)

            with st.expander("Ver forecast diario detallado"):
                st.dataframe(forecast_df[['fecha', 'forecast']].rename(columns={
                    'fecha': 'Fecha',
                    'forecast': 'Forecast (unidades)'
                }), use_container_width=True)

        except FileNotFoundError:
            st.warning("Modelo no encontrado. Asegúrate de correr el notebook de modelado primero.")
    else:
        st.info(f"El modelo para el SKU {sku_sel} estará disponible próximamente.")

# PÁGINA 3: REABASTECIMIENTO — SHELSY

elif pagina == "Reabastecimiento":
    st.title("Reabastecimiento")
    st.info("Esta página está siendo desarrollada por Shelsy.")