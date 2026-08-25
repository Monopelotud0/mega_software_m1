"""Gestor de configuración centralizado basado en YAML.

Implementa el patrón Singleton para garantizar una única carga del archivo
``config/params.yaml`` durante toda la vida de la aplicación. Provee acceso
seguro a parámetros anidados mediante rutas con punto, por ejemplo
``statistics.p_value_threshold``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.core.logger import setup_logger

logger = setup_logger(__name__)

#: Ruta por defecto al archivo de parámetros.
DEFAULT_PARAMS_PATH: Path = Path("config") / "params.yaml"

#: Tipos atómicos soportados como valor final de un parámetro.
ParamValue = Union[str, int, float, bool, None]


class ConfigManager:
    """Singleton que carga y expone la configuración YAML del proyecto.

    Atributos:
        _instance: Única instancia de la clase.
        _config: Diccionario con la configuración cargada.
        _loaded: Indica si la configuración ya fue cargada exitosamente.

    Ejemplo:
        >>> cfg = ConfigManager()
        >>> cfg.get_param("statistics.p_value_threshold")
        0.05
    """

    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}
    _loaded: bool = False

    def __new__(cls, params_path: Optional[Union[str, Path]] = None) -> "ConfigManager":
        """Garantiza que solo exista una instancia de ``ConfigManager``.

        Args:
            params_path: Ruta opcional al archivo YAML. En la primera
                instanciación se usa para cargar la configuración. Llamadas
                posteriores ignoran este argumento y devuelven la instancia
                existente.

        Returns:
            La instancia única de ``ConfigManager``.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(params_path or DEFAULT_PARAMS_PATH)
        return cls._instance

    def _load(self, params_path: Union[str, Path]) -> None:
        """Carga el archivo YAML en la memoria interna.

        Args:
            params_path: Ruta al archivo de configuración.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            yaml.YAMLError: Si el contenido no es YAML válido.
        """
        path = Path(params_path)
        logger.debug("Cargando configuración desde %s", path.resolve())

        if not path.is_file():
            msg = f"El archivo de configuración no existe: {path.resolve()}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            msg = f"El archivo YAML está mal formado: {path.resolve()}"
            logger.error(msg)
            raise yaml.YAMLError(msg) from exc

        if not isinstance(data, dict):
            msg = f"El archivo YAML debe contener un diccionario raíz: {path.resolve()}"
            logger.error(msg)
            raise ValueError(msg)

        ConfigManager._config = data
        ConfigManager._loaded = True
        logger.info("Configuración cargada exitosamente desde %s", path.resolve())

    @classmethod
    def reset(cls) -> None:
        """Reinicia el singleton (principalmente útil para tests).

        Fuerza la recreación de la instancia en la siguiente llamada a
        ``ConfigManager()``.
        """
        cls._instance = None
        cls._config = {}
        cls._loaded = False
        logger.debug("ConfigManager reiniciado")

    def get_param(self, key_path: str, default: Any = None) -> Any:
        """Obtiene un parámetro de configuración usando notación de puntos.

        Navega recursivamente por el diccionario interno siguiendo las claves
        separadas por ``.``. Si alguna clave intermedia no existe o no es un
        diccionario, se devuelve el valor por defecto.

        Args:
            key_path: Ruta al parámetro, por ejemplo ``"nlp.umap_components"``.
            default: Valor a devolver si el parámetro no existe. Por defecto
                es ``None``.

        Returns:
            Valor del parámetro o ``default`` si no se encuentra.

        Raises:
            RuntimeError: Si se intenta acceder antes de cargar la configuración.
        """
        if not ConfigManager._loaded:
            msg = "La configuración no ha sido cargada todavía."
            logger.error(msg)
            raise RuntimeError(msg)

        keys = key_path.split(".")
        value: Any = ConfigManager._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logger.debug("Parámetro no encontrado: %s; usando default", key_path)
                return default

        return value

    def get_system_param(self, key: str, default: Any = None) -> Any:
        """Acceso directo a la sección ``system``.

        Args:
            key: Clave dentro de la sección ``system``.
            default: Valor por defecto si no existe.

        Returns:
            Valor del parámetro o ``default``.
        """
        return self.get_param(f"system.{key}", default)

    def get_statistics_param(self, key: str, default: Any = None) -> Any:
        """Acceso directo a la sección ``statistics``.

        Args:
            key: Clave dentro de la sección ``statistics``.
            default: Valor por defecto si no existe.

        Returns:
            Valor del parámetro o ``default``.
        """
        return self.get_param(f"statistics.{key}", default)

    def get_nlp_param(self, key: str, default: Any = None) -> Any:
        """Acceso directo a la sección ``nlp``.

        Args:
            key: Clave dentro de la sección ``nlp``.
            default: Valor por defecto si no existe.

        Returns:
            Valor del parámetro o ``default``.
        """
        return self.get_param(f"nlp.{key}", default)

    @property
    def config(self) -> Dict[str, Any]:
        """Devuelve una copia superficial de la configuración cargada."""
        return ConfigManager._config.copy()

    @property
    def loaded(self) -> bool:
        """Indica si la configuración fue cargada exitosamente."""
        return ConfigManager._loaded
