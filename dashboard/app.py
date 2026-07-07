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

    sku_sel = st.selectbox("SKU", options=list(SKUS.keys()),
                           format_func=lambda x: f"{x} — {SKUS[x]}")

    @st.cache_data
    def cargar_ventas(sku):
        df = pd.read_csv(f"DATA/raw/ventashistoricas_{sku}.csv")
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df = df[df['Quantity'] > 0].dropna(subset=['Quantity'])
        diaria = df.groupby('Transaction_Date')['Quantity'].sum().reset_index()
        diaria.columns = ['fecha', 'demanda']
        return diaria.sort_values('fecha').reset_index(drop=True)

    df_ventas = cargar_ventas(sku_sel)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demanda promedio", f"{df_ventas['demanda'].mean():.0f} u/día")
    col2.metric("Demanda máxima", f"{df_ventas['demanda'].max():.0f} u")
    col3.metric("Días de historial", f"{len(df_ventas):,}")
    col4.metric("Último dato", df_ventas['fecha'].max().strftime("%d/%m/%Y"))

    st.divider()

    st.subheader("Demanda histórica (mensual)")
    mensual = df_ventas.set_index('fecha')['demanda'].resample('MS').sum()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mensual.index, mensual.values, linewidth=1.2)
    ax.set_ylabel("Unidades / mes")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    st.subheader("Forecast próximos 30 días")

    if sku_sel == '19127':
        try:
            modelo = joblib.load("models/modelo_19127.pkl")

            historial = df_ventas['demanda'].values.tolist()
            ultima_fecha = df_ventas['fecha'].max()
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

            col1, col2 = st.columns(2)
            col1.metric("Forecast total 30 días",
                        f"{forecast_df['forecast'].sum():,} unidades")
            col2.metric("Forecast diario promedio",
                        f"{forecast_df['forecast'].mean():.0f} unidades")

            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ultimos_60 = df_ventas.tail(60)
            ax2.plot(ultimos_60['fecha'], ultimos_60['demanda'],
                     label='Histórico', linewidth=1.2, color='#1f77b4')
            ax2.plot(forecast_df['fecha'], forecast_df['forecast'],
                     label='Forecast', linewidth=1.2,
                     color='#ff7f0e', linestyle='--')
            ax2.axvline(ultima_fecha, color='gray',
                        linestyle=':', linewidth=1)
            ax2.set_ylabel("Unidades / día")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

            st.dataframe(forecast_df.rename(columns={
                'fecha': 'Fecha',
                'forecast': 'Forecast (unidades)'
            }), use_container_width=True)

        except FileNotFoundError:
            st.warning("Modelo no encontrado.")
    else:
        st.info(f"El modelo para el SKU {sku_sel} estará disponible próximamente.")

# PÁGINA 3: REABASTECIMIENTO — SHELSY

elif pagina == "Reabastecimiento":
    st.title("Reabastecimiento")
    st.info("Esta página está siendo desarrollada por Shelsy.")