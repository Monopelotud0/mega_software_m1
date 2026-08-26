"""Orquestador principal del Mega Software de ML.

Este módulo es la puerta de entrada al sistema. Coordina la ejecución de las
fases automatizadas del pipeline:

1. Fase 0 - Ingesta y ruteo automático.
2. Fase 1 - Aprendizaje supervisado (clasificación/regresión).
3. Fase 2 - Aprendizaje no supervisado (clustering).
4. Fase 3 - Procesamiento de Lenguaje Natural (NLP / topic modeling).

Uso típico::

    python main.py data/01_raw/mi_dataset.csv --ai
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

from src.core.config import ConfigManager
from src.core.logger import setup_logger
from src.pipelines.fase_0_ingest import DataIngestionPipeline
from src.pipelines.fase_1_superv import SupervisedPipeline
from src.pipelines.fase_2_unsuperv import UnsupervisedPipeline
from src.pipelines.fase_3_nlp import NLPPipeline


def _parse_args() -> argparse.Namespace:
    """Configura y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="MegaSoftwareML",
        description="Orquestador del Mega Software de Machine Learning.",
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Ruta al archivo de datos (CSV, Excel o Parquet).",
    )
    # Se remueve el flag --ai para que se pregunte en tiempo de ejecución
    return parser.parse_args()

def _load_env() -> None:
    """Carga manualmente las variables del archivo .env si existe."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()


def _build_pipeline_id(filepath: str) -> str:
    """Genera un pipeline_id a partir del nombre base del archivo."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    return base_name.strip().replace(" ", "_") or "main_pipeline"


def _ask_task_type() -> str:
    """Solicita al usuario el tipo de tarea supervisada."""
    prompt = (
        "Seleccione el tipo de tarea supervisada:\n"
        "  1) Clasificación\n"
        "  2) Regresión\n"
        "Opción [1/2]: "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"1", "classification", "clasificacion", "c"}:
            return "classification"
        if choice in {"2", "regression", "regresion", "r"}:
            return "regression"
        print("Opción no válida. Intente nuevamente.")

def _ask_ai_mode() -> bool:
    """Consulta si se desea habilitar el uso de IA (Kimi)."""
    prompt = "¿Deseas habilitar el procesamiento con Inteligencia Artificial (Kimi)? (Selecciona 'n' para máxima privacidad / Zero-Trust) [s/n]: "
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"s", "si", "sí", "y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Opción no válida. Intente nuevamente.")


def _ask_text_column() -> str:
    """Solicita al usuario el nombre de la columna de texto."""
    return input("Ingrese el nombre exacto de la columna de texto: ").strip()


def _ask_autonomous_mode() -> bool:
    """Consulta si se desea modo autónomo."""
    prompt = "¿Desea ejecutar en modo 100% autónomo (sin interrupciones) o interactivo? [a/i]: "
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"a", "autonomo", "autónomo"}:
            return True
        if choice in {"i", "interactivo"}:
            return False
        print("Opción no válida. Intente nuevamente.")


def main() -> None:
    """Punto de entrada principal del orquestador."""
    _load_env()
    args = _parse_args()
    logger = setup_logger("MainOrchestrator")
    config = ConfigManager()

    try:
        logger.info("=" * 60)
        logger.info("Iniciando Mega Software de ML")
        logger.info("Archivo: %s", args.filepath)
        logger.info("=" * 60)

        ai_mode = _ask_ai_mode()
        is_autonomous = _ask_autonomous_mode()
        
        logger.info("Modo IA: %s | Modo de ejecución: %s", "Habilitado" if ai_mode else "Deshabilitado (Zero-Trust)", "Autónomo" if is_autonomous else "Interactivo")

        # ------------------------------------------------------------------
        # Paso 1: Fase 0 - Ingesta, limpieza y determinación de ruta
        # ------------------------------------------------------------------
        pipeline_id = _build_pipeline_id(args.filepath)
        ingestion = DataIngestionPipeline(pipeline_id=pipeline_id)
        route, target_col = ingestion.execute(filepath=args.filepath, ai_mode=ai_mode, autonomous=is_autonomous)

        logger.info("Ruta determinada por Fase 0: %s | target: %s", route, target_col)

        # ------------------------------------------------------------------
        # Paso 2: Enrutamiento dinámico a la fase correspondiente
        # ------------------------------------------------------------------
        if route == "fase_1":
            logger.info("Enrutando a Fase 1 - Supervisado")

            if is_autonomous:
                import json
                meta_path = os.path.join("data", "02_checkpoints", f"{pipeline_id}_fase_0_metadata.json")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    cardinality = meta.get("columns_metadata", {}).get(target_col, {}).get("cardinalidad", 0)
                    task_type = "classification" if cardinality < 15 else "regression"
                    logger.info("Modo autónomo: task_type inferido como '%s' (cardinalidad=%s)", task_type, cardinality)
                except Exception as exc:
                    logger.warning("No se pudo inferir task_type, asumiendo classification: %s", exc)
                    task_type = "classification"
            else:
                task_type = _ask_task_type()
            
            supervised = SupervisedPipeline(pipeline_id=pipeline_id)
            result = supervised.execute(target_col=target_col, task_type=task_type)
            logger.info("Resultado Fase 1: modelo=%s | métricas=%s", result.get("selected_model"), result.get("test_metrics"))

        elif route == "fase_2":
            logger.info("Enrutando a Fase 2 - No Supervisado")
            unsupervised = UnsupervisedPipeline(pipeline_id=pipeline_id)
            unsupervised.execute()

        elif route == "fase_3":
            logger.info("Enrutando a Fase 3 - NLP")
            
            if is_autonomous:
                import json
                meta_path = os.path.join("data", "02_checkpoints", f"{pipeline_id}_fase_0_metadata.json")
                text_col = None
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    for col, col_meta in meta.get("columns_metadata", {}).items():
                        if col_meta.get("tipo") == "Texto Libre":
                            text_col = col
                            break
                    if not text_col:
                        text_col = list(meta.get("columns_metadata", {}).keys())[0]
                    logger.info("Modo autónomo: text_col inferido como '%s'", text_col)
                except Exception as exc:
                    logger.warning("No se pudo inferir text_col: %s", exc)
                    text_col = "Var_01"
            else:
                text_col = _ask_text_column()

            nlp = NLPPipeline(pipeline_id=pipeline_id)
            result = nlp.execute(text_col=text_col, target_col=target_col or None)
            logger.info("Resultado Fase 3: tópicos=%s", result.get("metadata", {}).get("n_topics"))

        else:
            raise ValueError(f"Ruta desconocida devuelta por Fase 0: {route}")

        logger.info("Pipeline ejecutado con éxito. Checkpoints y modelos guardados.")

    except Exception as exc:
        logger.critical("Error crítico en la ejecución del pipeline: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
