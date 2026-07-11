import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import timedelta

# CONFIGURACIÓN
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "Mercalia — Planeación de Demanda e Inventario"

SKUS = {
    '19127': 'Duracell Batteries AA 40pk',
    '294000': 'Storage Box 27gal Professional',
    '455151': 'MS 5-tier Storage Rack'
}
COLORES = {'19127': '#1f77b4', '294000': '#ff7f0e', '455151': '#2ca02c'}
MAPE_MODELOS = {'19127': 24.10, '294000': None, '455151': 49.28}
MODELOS = {
    '19127': 'models/modelo_19127_colombia.pkl',
    '294000': 'models/modelo_294000_Colombia.pkl',
    '455151': 'models/modelo_455151_colombia.pkl'
}

# FUNCIONES DE DATOS
def cargar_ventas(sku, pais='Colombia'):
    df = pd.read_csv(f"DATA/raw/ventashistoricas_{sku}.csv")
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
    df = df[df['Quantity'] > 0].dropna(subset=['Quantity'])
    if pais != 'Todos los países':
        df = df[df['Country'] == pais]
    diaria = df.groupby('Transaction_Date')['Quantity'].sum().reset_index()
    diaria.columns = ['fecha', 'demanda']
    return diaria.sort_values('fecha').reset_index(drop=True)

def get_paises(sku):
    df = pd.read_csv(f"DATA/raw/ventashistoricas_{sku}.csv")
    return ['Todos los países'] + sorted(df['Country'].dropna().unique().tolist())

def hacer_forecast(sku, n_dias=30):
    try:
        modelo = joblib.load(MODELOS[sku])
        df = cargar_ventas(sku, 'Colombia')
        historial = df['demanda'].values.tolist()
        ultima_fecha = df['fecha'].max()
        fechas_futuras = pd.date_range(
            start=ultima_fecha + timedelta(days=1), periods=n_dias)
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
        return pd.DataFrame({
            'fecha': fechas_futuras,
            'forecast': [round(p) for p in predicciones]
        })
    except FileNotFoundError:
        return None

# LAYOUT
SIDEBAR = dbc.Col([
    html.Div([
        html.H4("📦 Mercalia", className="text-white fw-bold mb-4"),
        dbc.Nav([
            dbc.NavLink("📊 Estatus general",    href="/estatus",          active="exact", className="text-white mb-2"),
            dbc.NavLink("📈 Demanda y forecast", href="/demanda",          active="exact", className="text-white mb-2"),
            dbc.NavLink("🚚 Reabastecimiento",   href="/reabastecimiento", active="exact", className="text-white mb-2"),
        ], vertical=True),
    ], className="p-3 h-100")
], width=2, style={"backgroundColor": "#1a3a5c", "minHeight": "100vh"})

app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    dbc.Row([
        SIDEBAR,
        dbc.Col(html.Div(id="contenido-pagina"), width=10, className="p-4")
    ])
], fluid=True)

# PÁGINA 1: ESTATUS GENERAL — ALE
layout_estatus = html.Div([
    html.H2("Estatus general"),
    dbc.Alert("Esta página está siendo desarrollada por Ale. 🔧", color="warning")
])

# PÁGINA 2: DEMANDA Y FORECAST — GABI
layout_demanda = html.Div([
    html.H2("Demanda y forecast", className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("SKU"),
            dcc.Dropdown(
                id="sku-selector",
                options=[{"label": f"{k} — {v}", "value": k} for k, v in SKUS.items()],
                value="19127", clearable=False
            )
        ], width=5),
        dbc.Col([
            html.Label("País"),
            dcc.Dropdown(id="pais-selector", value="Colombia", clearable=False)
        ], width=3),
        dbc.Col([
            html.Label("Horizonte forecast (días)"),
            dcc.Slider(id="horizonte-slider", min=7, max=90, step=7,
                       value=30, marks={7:"7d", 30:"30d", 60:"60d", 90:"90d"})
        ], width=4),
    ], className="mb-4"),

    dbc.Row(id="metricas-cards", className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Demanda histórica y forecast"),
        dbc.CardBody(dcc.Graph(id="grafica-forecast"))
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Forecast semanal"),
            dbc.CardBody(html.Div(id="tabla-forecast-semanal"))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Estacionalidad por mes"),
            dbc.CardBody(dcc.Graph(id="grafica-estacionalidad"))
        ]), width=6),
    ])
])

# PÁGINA 3: REABASTECIMIENTO — SHELSY

def cargar_datos_reabastecimiento(sku):
    LEAD_TIMES = {'19127': 15.2, '294000': 36.3, '455151': 109.7}
    PO_FILES = {
        '19127': 'DATA/raw/POs_19127.csv',
        '294000': 'DATA/raw/POs_2974000.csv',
        '455151': 'DATA/raw/POs_455151.csv'
    }
    inv = pd.read_csv('DATA/raw/Inventario actual.csv', sep=';',
                      skiprows=2, encoding='utf-8-sig')
    inv['In Loc Days of Supply'] = pd.to_numeric(
        inv['In Loc Days of Supply'], errors='coerce')
    inv['FCST Next 30 Days'] = pd.to_numeric(
        inv['FCST Next 30 Days'], errors='coerce')

    sub = inv[inv['Item number'].astype(str).str.strip() == sku]
    inv_actual = sub['End balance'].sum()
    dias_stock = sub[sub['In Loc Days of Supply'] < 999]['In Loc Days of Supply'].mean()
    fcst_horizonte = sub['FCST Next 30 Days'].sum()

    po = pd.read_csv(PO_FILES[sku], parse_dates=['Revised_Delivery_Date'])
    abiertas = po[po['Status'] == 'Open']
    pos_camino = abiertas['Quantity_Open'].sum()
    prox_entrega = abiertas['Revised_Delivery_Date'].min() if len(abiertas) else None

    lt = LEAD_TIMES[sku]
    demanda_diaria = fcst_horizonte / 30 if fcst_horizonte else 1
    fecha_quiebre = pd.Timestamp.now() + pd.Timedelta(days=dias_stock) if dias_stock else None

    if dias_stock < lt:
        estado = 'Reabastecer'
        color_estado = 'danger'
    elif dias_stock < lt * 1.5:
        estado = 'Monitorear'
        color_estado = 'warning'
    else:
        estado = 'Normal'
        color_estado = 'success'

    return {
        'inv_actual': inv_actual,
        'fcst_horizonte': fcst_horizonte,
        'pos_camino': pos_camino,
        'dias_stock': dias_stock,
        'fecha_quiebre': fecha_quiebre,
        'estado': estado,
        'color_estado': color_estado,
        'abiertas': abiertas,
        'demanda_diaria': demanda_diaria,
        'lead_time': lt
    }

layout_reabastecimiento = html.Div([
    html.H2("Reabastecimiento", className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("SKU"),
            dcc.Dropdown(
                id="sku-reab",
                options=[{"label": f"{k} — {v}", "value": k} for k, v in SKUS.items()],
                value="455151", clearable=False
            )
        ], width=4),
        dbc.Col([
            html.Label("Horizonte (días)"),
            dcc.Slider(id="horizonte-reab", min=7, max=90, step=7,
                       value=30, marks={7:"7d", 30:"30d", 60:"60d", 90:"90d"})
        ], width=4),
    ], className="mb-4"),

    dbc.Row(id="kpis-reab", className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Túnel de reabastecimiento"),
        dbc.CardBody(dcc.Graph(id="grafica-reab"))
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Próximas recepciones"),
        dbc.CardBody(html.Div(id="tabla-reab"))
    ])
])

@app.callback(
    Output("kpis-reab", "children"),
    Output("grafica-reab", "figure"),
    Output("tabla-reab", "children"),
    Input("sku-reab", "value"),
    Input("horizonte-reab", "value"),
)
def actualizar_reabastecimiento(sku, horizonte):
    if not sku:
        return [], go.Figure(), html.Div()

    datos = cargar_datos_reabastecimiento(sku)

    # KPIs
    kpis = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Inventario actual", className="text-muted small mb-1"),
            html.H4(f"{datos['inv_actual']:,.0f} u", className="fw-bold")
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Forecast horizonte", className="text-muted small mb-1"),
            html.H4(f"{datos['fcst_horizonte']:,.0f} u", className="fw-bold")
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("POs en camino", className="text-muted small mb-1"),
            html.H4(f"{datos['pos_camino']:,.0f} u", className="fw-bold")
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Días de stock", className="text-muted small mb-1"),
            html.H4(f"{datos['dias_stock']:.1f} días", className="fw-bold")
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Fecha posible de quiebre", className="text-muted small mb-1"),
            html.H4(datos['fecha_quiebre'].strftime('%d/%m/%Y') if datos['fecha_quiebre'] else '—',
                    className="fw-bold text-danger")
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Estado", className="text-muted small mb-1"),
            dbc.Badge(datos['estado'], color=datos['color_estado'],
                      className="px-3 py-2 fs-6")
        ])), width=2),
    ])

    # Túnel de reabastecimiento
    fechas = pd.date_range(start=pd.Timestamp.now(), periods=horizonte)
    inv_proyectado = [max(0, datos['inv_actual'] - datos['demanda_diaria'] * i)
                      for i in range(horizonte)]

    # Agregar impacto de POs
    abiertas = datos['abiertas'].copy()
    for i, fecha in enumerate(fechas):
        entregas = abiertas[abiertas['Revised_Delivery_Date'] <= fecha]['Quantity_Open'].sum()
        inv_proyectado[i] = max(0, inv_proyectado[i] + entregas)

    stock_min = datos['demanda_diaria'] * datos['lead_time']
    stock_max = datos['demanda_diaria'] * datos['lead_time'] * 3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fechas, y=inv_proyectado,
        name='Inventario proyectado',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fechas, y=[datos['demanda_diaria']] * horizonte,
        name='Forecast de demanda',
        line=dict(color='#ff7f0e', dash='dash', width=1.5)
    ))
    fig.add_hline(y=stock_min, line_dash="dot", line_color="red",
                  annotation_text="Stock mínimo")
    fig.add_hline(y=stock_max, line_dash="dot", line_color="green",
                  annotation_text="Stock máximo")
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="Unidades",
        plot_bgcolor='white', paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    # Tabla próximas recepciones
    if len(abiertas) > 0:
        filas = []
        for _, row in abiertas.iterrows():
            dias_cobertura = row['Quantity_Open'] / datos['demanda_diaria'] if datos['demanda_diaria'] else 0
            filas.append(html.Tr([
                html.Td(row.get('Revised_Delivery_Date', '—').strftime('%d/%m/%Y')
                        if pd.notna(row.get('Revised_Delivery_Date')) else '—'),
                html.Td(str(row.get('Purchase_Order', '—'))),
                html.Td(f"{row['Quantity_Open']:,.0f}", className="text-center"),
                html.Td(f"{dias_cobertura:.1f} días", className="text-center"),
                html.Td(row.get('Vendor Name', '—')),
            ]))
        tabla = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Fecha entrega"), html.Th("PO"),
                html.Th("Cantidad", className="text-center"),
                html.Th("Cobertura aportada", className="text-center"),
                html.Th("Proveedor")
            ], style={"backgroundColor": "#1a3a5c", "color": "white"})),
            html.Tbody(filas)
        ], bordered=True, hover=True, responsive=True, size="sm")
    else:
        tabla = dbc.Alert("No hay órdenes de compra abiertas para este SKU.", color="info")

    return kpis, fig, tabla

# ROUTING
@app.callback(Output("contenido-pagina", "children"), Input("url", "pathname"))
def render_pagina(pathname):
    if pathname == "/demanda":
        return layout_demanda
    elif pathname == "/reabastecimiento":
        return layout_reabastecimiento
    else:
        return layout_estatus

# CALLBACKS PÁGINA 2
@app.callback(Output("pais-selector", "options"), Input("sku-selector", "value"))
def actualizar_paises(sku):
    return [{"label": p, "value": p} for p in get_paises(sku)]

@app.callback(
    Output("metricas-cards", "children"),
    Output("grafica-forecast", "figure"),
    Output("tabla-forecast-semanal", "children"),
    Output("grafica-estacionalidad", "figure"),
    Input("sku-selector", "value"),
    Input("pais-selector", "value"),
    Input("horizonte-slider", "value"),
)
def actualizar_pagina_demanda(sku, pais, horizonte):
    if not sku or not pais:
        return [], go.Figure(), html.Div(), go.Figure()

    color = COLORES.get(sku, "#1f77b4")
    df = cargar_ventas(sku, pais)
    mape = MAPE_MODELOS.get(sku)

    # Métricas
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Demanda promedio", className="text-muted mb-1 small"),
            html.H4(f"{df['demanda'].mean():.0f} u/día", className="fw-bold")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Demanda máxima", className="text-muted mb-1 small"),
            html.H4(f"{df['demanda'].max():.0f} u", className="fw-bold")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Días de historial", className="text-muted mb-1 small"),
            html.H4(f"{len(df):,}", className="fw-bold")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("MAPE modelo", className="text-muted mb-1 small"),
            html.H4(f"{mape:.1f}%" if mape else "Pendiente",
                    className="fw-bold text-success" if mape else "fw-bold text-muted")
        ])), width=3),
    ])

    # Gráfica forecast
    fig = go.Figure()
    ultimos_90 = df.tail(90)
    fig.add_trace(go.Scatter(
        x=ultimos_90['fecha'], y=ultimos_90['demanda'],
        name='Histórico', line=dict(color=color, width=1.5)
    ))

    forecast_df = hacer_forecast(sku, horizonte)
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df['fecha'], y=forecast_df['forecast'],
            name='Forecast XGBoost',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['fecha'], forecast_df['fecha'][::-1]]),
            y=pd.concat([forecast_df['forecast']*1.15, forecast_df['forecast'][::-1]*0.85]),
            fill='toself', fillcolor='rgba(255,127,14,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Banda ±15%'
        ))
        fig.add_vline(x=str(df['fecha'].max()), line_dash="dot", line_color="gray")
    else:
        fig.add_annotation(text="Modelo pendiente para este SKU",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="gray"))

    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="Unidades / día",
        plot_bgcolor='white', paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    # Tabla semanal
    if forecast_df is not None:
        forecast_df['semana'] = forecast_df['fecha'].dt.to_period('W').astype(str)
        semanal = forecast_df.groupby('semana')['forecast'].sum().reset_index()
        semanal.columns = ['Semana', 'Forecast (u)']
        tabla = dash_table.DataTable(
            data=semanal.to_dict('records'),
            columns=[{"name": c, "id": c} for c in semanal.columns],
            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 13},
            style_header={'backgroundColor': '#1a3a5c', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}]
        )
    else:
        tabla = dbc.Alert("Forecast disponible próximamente.", color="info")

    # Estacionalidad
    df_mes = df.copy()
    df_mes['mes'] = df_mes['fecha'].dt.month
    nombres_meses = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
                     7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
    est = df_mes.groupby('mes')['demanda'].mean().reset_index()
    est['mes_nombre'] = est['mes'].map(nombres_meses)
    fig_est = go.Figure(go.Bar(
        x=est['mes_nombre'], y=est['demanda'],
        marker_color=color, opacity=0.8
    ))
    fig_est.update_layout(
        height=250, margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Demanda promedio",
        plot_bgcolor='white', paper_bgcolor='white'
    )
    fig_est.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    return cards, fig, tabla, fig_est

if __name__ == "__main__":
    app.run(debug=True)