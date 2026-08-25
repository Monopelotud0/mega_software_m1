# Mega Software ML

Orquestador de Machine Learning autónomo, híbrido y de alta privacidad (Zero Trust) diseñado para ejecutarse localmente y procesar grandes volúmenes de datos.

## 🚀 Arquitectura y Fases del Pipeline

El sistema está estructurado en 4 fases principales gestionadas por el `MainOrchestrator`:

1. **Fase 0 - Ingesta y Limpieza:** Carga archivos (CSV, Excel, Parquet), realiza limpiezas automáticas (con soporte opcional de IA vía LLM) y determina de manera inteligente la ruta (el tipo de modelo a entrenar).
2. **Fase 1 - Supervisado:** Ejecuta modelos de Clasificación o Regresión en base a la ruta establecida y guarda el mejor modelo.
3. **Fase 2 - No Supervisado:** Realiza Clustering automatizado de los datos.
4. **Fase 3 - NLP (Procesamiento de Lenguaje Natural):** Ejecuta vectorizaciones avanzadas (TF-IDF, Embeddings densos con SentenceTransformer), topic modeling (UMAP + HDBSCAN) sobre campos de texto libre.

## 🛠️ Instalación y Requisitos

Este proyecto utiliza [Poetry](https://python-poetry.org/) como gestor de dependencias (Python >= 3.10).

1. Asegúrate de tener Poetry instalado.
2. En la raíz del proyecto, instala las dependencias (se creará el entorno virtual `venv` o `.venv` automáticamente):
   ```bash
   poetry install
   ```

## ⚙️ Uso

El punto de entrada principal es `main.py`. Requiere que proporciones la ruta de tu dataset.

```bash
# Ejecución básica (modo interactivo por defecto)
python main.py data/01_raw/mi_dataset.csv

# Ejecución activando el soporte de IA para limpieza
python main.py data/01_raw/mi_dataset.csv --ai
```

### Modos de Ejecución
Al iniciar el orquestador, se te preguntará si deseas utilizar el **Modo Autónomo** (ejecución sin interrupciones, infiriendo automáticamente variables y tipos de tareas basado en la Fase 0) o el **Modo Interactivo** (solicita *inputs* manuales del usuario cuando es necesario).

## 🔒 Privacidad (Zero Trust)
Todos los datos, modelos (incluyendo LLMs y embeddings locales) y checkpoints se procesan localmente sin enviar datos sensibles a servicios de terceros, preservando la confidencialidad absoluta del dataset de entrada.
