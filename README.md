# Mega Software ML

Orquestador de Machine Learning autónomo y de alta privacidad (Zero Trust) diseñado para ejecutarse localmente y procesar grandes volúmenes de datos.

## 🚀 Arquitectura y Fases del Pipeline

El sistema está estructurado en 4 fases principales gestionadas por el `MainOrchestrator`:

1. **Fase 0 - Ingesta y Limpieza:** Carga archivos (CSV, Excel, Parquet), extrae metadatos y realiza limpiezas automáticas. Puede usar reglas deterministas o integrarse opcionalmente con **Inteligencia Artificial (Kimi-k2.6 vía Moonshot)** para tomar decisiones de preprocesamiento avanzadas sin comprometer los datos reales.
2. **Fase 1 - Supervisado:** Ejecuta modelos de Clasificación o Regresión en base a la ruta establecida y guarda el mejor modelo.
3. **Fase 2 - No Supervisado:** Realiza Clustering automatizado de los datos.
4. **Fase 3 - NLP (Procesamiento de Lenguaje Natural):** Ejecuta vectorizaciones avanzadas (TF-IDF, Embeddings densos con SentenceTransformer), topic modeling (UMAP + HDBSCAN) sobre campos de texto libre.

## 🛠️ Instalación y Requisitos

Este proyecto gestiona sus dependencias (Python >= 3.10) a través de `pyproject.toml`.

1. Crea y activa tu entorno virtual (por ejemplo, con `uv` o `venv` tradicional).
2. Instala las dependencias:
   ```bash
   poetry install
   # O si usas pip estándar:
   pip install .
   ```

### Configuración del Entorno (.env)
Para habilitar la Inteligencia Artificial, debes crear un archivo `.env` en la raíz del proyecto guardado con formato `UTF-8` y agregar tu clave de API:
```env
MOONSHOT_API_KEY=tu_clave_secreta_aqui
```

## ⚙️ Uso

El punto de entrada principal es `main.py`. Requiere que proporciones la ruta de tu dataset por consola.

```bash
python main.py data/01_raw/mi_dataset.csv
```

### Opciones de Ejecución Interactiva
Al iniciar el orquestador, el sistema te hará dos preguntas en pantalla (ya no se usan argumentos obsoletos como `--ai`):
1. **Inteligencia Artificial (Kimi):** Selecciona `s` para usar el LLM de Moonshot como analista de datos, o `n` para aplicar reglas deterministas 100% offline.
2. **Autonomía:** Selecciona `a` para que el software no se detenga e infiera todo automáticamente, o `i` para que te vaya pidiendo confirmaciones manuales.

## 🔒 Privacidad (Zero Trust Parcial con IA)
Todos los datos pesados, modelos pesados y checkpoints se procesan localmente sin enviar tus datos a terceros.
Si habilitas el uso de IA, el sistema aplica un enmascaramiento estricto: **sólo se envían resúmenes matemáticos (metadatos) con nombres de columnas ocultos (Var_01, Var_02...)**. Las filas, nombres y datos reales jamás salen de tu disco duro, protegiendo al máximo tu información confidencial.
