"""Clase base abstracta para todos los pipelines del Mega Software de ML.

Define la interfaz común que debe implementar cada fase: inicialización con
configuración, logger y gestor de checkpoints, además de un mecanismo seguro
para ejecutar pasos críticos mediante ``safe_execute``.
"""

from __future__ import annotations

import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd

from src.core.config import ConfigManager
from src.core.logger import setup_logger
from src.core.state_machine import CheckpointManager

T = TypeVar("T")


class BasePipeline(ABC):
    """Pipeline base con servicios compartidos del sistema.

    Cada pipeline concreto hereda de esta clase y obtiene acceso al gestor de
    configuración, logger y checkpointing. También dispone del método
    ``safe_execute`` para envolver operaciones críticas con manejo uniforme de
    errores.

    Args:
        pipeline_id: Identificador único de la ejecución del pipeline.
        params_path: Ruta opcional al archivo de configuración YAML.

    Attributes:
        pipeline_id: Identificador de la ejecución.
        config: Instancia singleton de ``ConfigManager``.
        logger: Logger configurado para el pipeline.
        checkpoint_manager: Gestor de checkpoints en disco.
        df: ``DataFrame`` en memoria (inicialmente ``None``).
        metadata: Metadatos acumulados de la fase.
    """

    def __init__(
        self,
        pipeline_id: str,
        params_path: Optional[str] = None,
    ) -> None:
        """Inicializa los servicios compartidos del pipeline."""
        if not isinstance(pipeline_id, str) or not pipeline_id.strip():
            raise ValueError("pipeline_id debe ser un string no vacío")

        self.pipeline_id: str = pipeline_id.strip()
        self.config: ConfigManager = ConfigManager(params_path)
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}.{pipeline_id}")
        self.checkpoint_manager: CheckpointManager = CheckpointManager(self.pipeline_id)

        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

        self.logger.info("Pipeline inicializado: %s", self.pipeline_id)

    def safe_execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Ejecuta una función envuelta en manejo de excepciones.

        Si la función lanza una excepción, se registra el error completo y se
        relanza para detener el pipeline de forma controlada.

        Args:
            func: Función a ejecutar.
            *args: Argumentos posicionales para ``func``.
            **kwargs: Argumentos nombrados para ``func``.

        Returns:
            El valor retornado por ``func``.

        Raises:
            Exception: Propaga cualquier excepción lanzada por ``func``.
        """
        try:
            self.logger.debug("Ejecutando %s", func.__name__)
            return func(*args, **kwargs)
        except Exception as exc:
            self.logger.error(
                "Error en %s: %s\n%s",
                func.__name__,
                exc,
                traceback.format_exc(),
            )
            raise

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> None:
        """Punto de entrada principal del pipeline.

        Cada subclase debe implementar este método definiendo el flujo
        completo de su fase.
        """
        raise NotImplementedError("Las subclases deben implementar run()")
