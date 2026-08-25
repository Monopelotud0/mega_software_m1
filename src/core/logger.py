"""Módulo de logging centralizado para el Mega Software de ML.

Este módulo expone la función [`setup_logger`](./src/core/logger.py:52) que configura
loggers con salida dual:

- Consola (nivel INFO) para supervisión operativa.
- Archivo ``logs/pipeline.log`` (nivel DEBUG) para trazabilidad completa.

Todos los loggers comparten el mismo formateador profesional y no propagan
mensajes al logger raíz para evitar duplicados.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

#: Formato profesional: timestamp | nivel | nombre | mensaje
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Fecha/Hora en formato ISO 8601 legible.
DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"

#: Directorio y archivo de logs por defecto.
LOG_DIR: Final[Path] = Path("logs")
LOG_FILE: Final[Path] = LOG_DIR / "pipeline.log"

#: Nivel de logging para consola y archivo.
CONSOLE_LEVEL: Final[int] = logging.INFO
FILE_LEVEL: Final[int] = logging.DEBUG


def _ensure_log_directory() -> None:
    """Crea el directorio de logs si no existe.

    Es seguro llamarla múltiples veces porque ``Path.mkdir`` con
    ``exist_ok=True`` no genera errores si el directorio ya existe.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    """Configura y devuelve un logger con salida dual.

    El logger devuelto envía mensajes a la consola (nivel INFO o superior) y
    al archivo ``logs/pipeline.log`` (nivel DEBUG o superior). Si el logger
    ya fue configurado previamente, se reutiliza la misma configuración para
    evitar handlers duplicados.

    Args:
        name: Nombre identificador del logger. Se recomienda usar ``__name__``.

    Returns:
        Logger configurado listo para usar.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Evitar duplicación si el logger ya fue configurado en esta ejecución.
    if logger.handlers:
        return logger

    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de archivo
    _ensure_log_directory()
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(FILE_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
