"""Máquina de Estados basada en checkpointing en disco.

Este módulo implementa la clase [`CheckpointManager`](./src/core/state_machine.py:24),
responsable de persistir el estado de cada fase del pipeline en archivos
``.parquet`` (datos) y ``.json`` (metadatos). Garantiza tolerancia a fallos
sin utilizar ``exec()`` ni manipulación dinámica de variables en RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid, ArrowException

from src.core.logger import setup_logger

logger = setup_logger(__name__)

#: Directorio base para checkpoints.
CHECKPOINT_DIR: Path = Path("data") / "02_checkpoints"


class CheckpointManager:
    """Gestiona la persistencia y recuperación de checkpoints por pipeline.

    Cada checkpoint se almacena como un archivo ``.parquet`` con el
    ``DataFrame`` de la fase y, opcionalmente, un archivo ``.json`` con
    metadatos asociados. Los nombres de archivo incluyen el ``pipeline_id``
    como prefijo para evitar colisiones entre ejecuciones.

    Args:
        pipeline_id: Identificador único del pipeline. Se usa como prefijo
            en los nombres de archivo.

    Attributes:
        pipeline_id: Identificador del pipeline.
        checkpoint_dir: Directorio donde se almacenan los checkpoints.
    """

    def __init__(self, pipeline_id: str) -> None:
        """Inicializa el gestor de checkpoints para un pipeline dado."""
        if not isinstance(pipeline_id, str) or not pipeline_id.strip():
            raise ValueError("pipeline_id debe ser un string no vacío")

        self.pipeline_id: str = pipeline_id.strip()
        self.checkpoint_dir: Path = CHECKPOINT_DIR
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self) -> None:
        """Crea el directorio de checkpoints si no existe."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Directorio de checkpoints listo: %s", self.checkpoint_dir.resolve())

    def _checkpoint_path(self, phase_name: str, extension: str) -> Path:
        """Construye la ruta de un archivo de checkpoint.

        Args:
            phase_name: Nombre de la fase del pipeline.
            extension: Extensión del archivo (incluyendo el punto).

        Returns:
            Ruta absoluta del archivo de checkpoint.
        """
        safe_phase = phase_name.strip().replace(" ", "_").lower()
        filename = f"{self.pipeline_id}_{safe_phase}{extension}"
        return self.checkpoint_dir / filename

    def save_checkpoint(
        self,
        phase_name: str,
        df: pd.DataFrame,
        metadata: Optional[Dict] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Persiste un ``DataFrame`` y metadatos opcionales en disco.

        El ``DataFrame`` se guarda en formato ``.parquet`` usando PyArrow como
        engine. Si se proporciona ``metadata``, se serializa como JSON en un
        archivo con el mismo prefijo que el parquet.

        Args:
            phase_name: Nombre de la fase del pipeline.
            df: ``DataFrame`` a persistir.
            metadata: Diccionario con metadatos asociados. Por defecto ``None``.

        Returns:
            Tupla con la ruta del archivo parquet y la ruta del archivo JSON
            (``None`` si no se proporcionaron metadatos).

        Raises:
            ValueError: Si ``phase_name`` es inválido o ``df`` no es un
                ``DataFrame``.
            IOError: Si ocurre un error de escritura en disco.
            ArrowException: Si PyArrow no puede convertir el ``DataFrame``.
        """
        if not isinstance(phase_name, str) or not phase_name.strip():
            raise ValueError("phase_name debe ser un string no vacío")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df debe ser una instancia de pd.DataFrame")

        parquet_path = self._checkpoint_path(phase_name, ".parquet")
        json_path: Optional[Path] = None

        try:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path)

            mem_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                "Checkpoint guardado: phase=%s | rows=%d | cols=%d | mem=%.2fMB | path=%s",
                phase_name,
                len(df),
                len(df.columns),
                mem_size_mb,
                parquet_path.resolve(),
            )

            if metadata is not None:
                json_path = self._checkpoint_path(phase_name, ".json")
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logger.info(
                    "Metadata guardada: phase=%s | path=%s",
                    phase_name,
                    json_path.resolve(),
                )

            return parquet_path, json_path

        except (ArrowInvalid, ArrowException) as exc:
            logger.error(
                "Error de PyArrow al guardar checkpoint %s: %s",
                phase_name,
                exc,
            )
            raise
        except OSError as exc:
            logger.error(
                "Error de escritura en disco para checkpoint %s: %s",
                phase_name,
                exc,
            )
            raise IOError(f"No se pudo escribir el checkpoint {phase_name}") from exc

    def load_checkpoint(self, phase_name: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Recupera un ``DataFrame`` y sus metadatos desde disco.

        Busca el archivo ``.parquet`` y el ``.json`` asociados a la fase. Si
        ambos existen, los carga y retorna. Si no existen, retorna
        ``(None, None)``.

        Args:
            phase_name: Nombre de la fase del pipeline.

        Returns:
            Tupla ``(DataFrame, metadata)``. Si no se encuentra el checkpoint,
            ambos valores son ``None``.

        Raises:
            ValueError: Si ``phase_name`` es inválido.
            IOError: Si ocurre un error de lectura en disco.
            ArrowException: Si el archivo parquet está corrupto o es inválido.
        """
        if not isinstance(phase_name, str) or not phase_name.strip():
            raise ValueError("phase_name debe ser un string no vacío")

        parquet_path = self._checkpoint_path(phase_name, ".parquet")
        json_path = self._checkpoint_path(phase_name, ".json")

        if not parquet_path.is_file():
            logger.debug("Checkpoint no encontrado: %s", parquet_path.resolve())
            return None, None

        df: Optional[pd.DataFrame] = None
        metadata: Optional[Dict] = None

        try:
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            logger.info(
                "Checkpoint cargado: phase=%s | rows=%d | cols=%d | path=%s",
                phase_name,
                len(df),
                len(df.columns),
                parquet_path.resolve(),
            )
        except (ArrowInvalid, ArrowException) as exc:
            logger.error(
                "Error de PyArrow al leer checkpoint %s: %s",
                phase_name,
                exc,
            )
            raise
        except OSError as exc:
            logger.error(
                "Error de lectura en disco para checkpoint %s: %s",
                phase_name,
                exc,
            )
            raise IOError(f"No se pudo leer el checkpoint {phase_name}") from exc

        if json_path.is_file():
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.debug(
                    "Metadata cargada: phase=%s | path=%s",
                    phase_name,
                    json_path.resolve(),
                )
            except json.JSONDecodeError as exc:
                logger.error(
                    "El archivo JSON de metadatos está corrupto %s: %s",
                    json_path.resolve(),
                    exc,
                )
                raise
            except OSError as exc:
                logger.error(
                    "Error de lectura de metadatos %s: %s",
                    json_path.resolve(),
                    exc,
                )
                raise IOError(f"No se pudo leer los metadatos de {phase_name}") from exc

        return df, metadata

    def list_checkpoints(self) -> list[str]:
        """Lista las fases que tienen al menos un checkpoint ``.parquet``.

        Returns:
            Lista de nombres de fase detectados en el directorio de
            checkpoints.
        """
        prefix = f"{self.pipeline_id}_"
        suffix = ".parquet"
        phases = []

        if not self.checkpoint_dir.exists():
            return phases

        for path in self.checkpoint_dir.glob(f"{prefix}*{suffix}"):
            phase_part = path.name[len(prefix) : -len(suffix)]
            phases.append(phase_part)

        return phases

    def clear_checkpoints(self) -> int:
        """Elimina todos los checkpoints asociados al ``pipeline_id`` actual.

        Returns:
            Cantidad de archivos eliminados.
        """
        prefix = f"{self.pipeline_id}_"
        removed = 0

        if not self.checkpoint_dir.exists():
            return removed

        for path in self.checkpoint_dir.glob(f"{prefix}*"):
            try:
                path.unlink()
                removed += 1
                logger.debug("Checkpoint eliminado: %s", path.resolve())
            except OSError as exc:
                logger.warning("No se pudo eliminar %s: %s", path.resolve(), exc)

        logger.info("Se eliminaron %d checkpoints para pipeline=%s", removed, self.pipeline_id)
        return removed
