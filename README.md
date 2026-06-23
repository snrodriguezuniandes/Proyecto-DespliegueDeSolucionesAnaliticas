# Planeación de Demanda e Inventario – Supermercado
 
Proyecto del curso **Despliegue de Soluciones Analíticas (MIIA-4304)** – Universidad de los Andes.
 
## Pregunta de negocio

¿Cómo apoyar la planeación de inventario y reabastecimiento mediante la anticipación demsobreinventario y quiebres de inventario, considerando demanda histórica, inventario disponible, órdenes de compra en camino y lead times de proveedores?
 
## Alcance

Prototipo funcional para 3 SKUs principales de replenishment:

- `19127` – Duracell Batteries AA 40pk

- `294000` – Storage Box 27gal Professional

- `455151` – MS 5-tier Storage Rack
 
## Estructura del repositorio

```

DATA/raw/          # datos originales (versionados con DVC)

DATA/processed/    # datos limpios/transformados

notebooks/         # exploración y prototipado

src/               # código (data, features, models, api)

dashboard/         # tablero

models/            # modelos entrenados

reports/figures/   # gráficas para el reporte

 