"""Fase 0: Ingesta de datos del Mega Software de ML.

Este módulo implementa la primera fase del pipeline, responsable de cargar
archivos de datos desde disco a memoria de forma robusta y defensiva.
Soporta formatos CSV, Excel y Parquet, e integra el checkpoint inicial en
``data/02_checkpoints/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.pipelines.base_pipeline import BasePipeline

# Imports para modo AI (lazy import dentro de orchestrate_cleaning)
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from src.schemas.llm_models import ColumnRecommendation, MetadataResponse
except ImportError:  # pragma: no cover
    MetadataResponse = None  # type: ignore
    ColumnRecommendation = None  # type: ignore


class DataIngestionPipeline(BasePipeline):
    """Pipeline de ingesta de datos con detección automática de formato.

    Hereda de ``BasePipeline`` y utiliza los servicios compartidos de
    configuración, logging y checkpointing. Carga archivos CSV, Excel o
    Parquet según su extensión, aplicando parámetros defensivos para
    maximizar la tasa de ingesta exitosa.

    Attributes:
        df: ``DataFrame`` cargado en memoria. ``None`` hasta que se ejecuta
            ``load_data``.
        column_mapping: Diccionario para el enmascaramiento futuro de nombres
            de columna. Inicialmente vacío.
    """

    def __init__(self, pipeline_id: str, params_path: Optional[str] = None) -> None:
        """Inicializa la fase de ingesta.

        Args:
            pipeline_id: Identificador único de la ejecución.
            params_path: Ruta opcional al archivo de configuración YAML.
        """
        super().__init__(pipeline_id, params_path)
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.logger.info("Fase 0 - Ingesta inicializada")

    def _check_memory_and_load(self, filepath: str) -> pd.DataFrame:
        """Carga un archivo de datos aplicando validaciones de memoria.

        Detecta el formato por la extensión, verifica el tamaño contra el
        umbral configurado y aplica lectura defensiva para CSV.

        Args:
            filepath: Ruta al archivo de datos.

        Returns:
            ``DataFrame`` cargado.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si la extensión no está soportada.
            pd.errors.EmptyDataError: Si el archivo no contiene datos.
            IOError: Si ocurre un error de lectura en disco.
        """
        path = Path(filepath)

        if not path.is_file():
            msg = f"El archivo no existe: {path.resolve()}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        file_size_mb = path.stat().st_size / (1024 * 1024)
        warning_threshold = self.config.get_param(
            "system.file_size_warning_mb",
            default=1500,
        )

        if file_size_mb > warning_threshold:
            self.logger.warning(
                "El archivo %.2f MB supera el umbral de advertencia de %s MB: %s",
                file_size_mb,
                warning_threshold,
                path.resolve(),
            )
        else:
            self.logger.info(
                "Tamaño del archivo %.2f MB dentro del límite (%s MB): %s",
                file_size_mb,
                warning_threshold,
                path.resolve(),
            )

        extension = path.suffix.lower()
        self.logger.info("Leyendo archivo %s con extensión %s", path.resolve(), extension)

        try:
            if extension == ".csv":
                df = pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",
                )
            elif extension in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            elif extension == ".parquet":
                df = pd.read_parquet(path)
            else:
                msg = f"Formato de archivo no soportado: {extension}"
                self.logger.error(msg)
                raise ValueError(msg)
        except pd.errors.EmptyDataError as exc:
            self.logger.error("El archivo está vacío: %s", path.resolve())
            raise
        except pd.errors.ParserError as exc:
            self.logger.error("Error al parsear el archivo %s: %s", path.resolve(), exc)
            raise
        except OSError as exc:
            self.logger.error("Error de lectura en disco %s: %s", path.resolve(), exc)
            raise IOError(f"No se pudo leer el archivo {path.resolve()}") from exc

        self.logger.info(
            "Archivo cargado exitosamente: rows=%d | cols=%d | path=%s",
            len(df),
            len(df.columns),
            path.resolve(),
        )
        return df

    def load_data(self, filepath: str) -> None:
        """Carga los datos de forma segura y persiste el checkpoint inicial.

        Envuelve ``_check_memory_and_load`` con ``safe_execute``, asigna el
        resultado a ``self.df`` y guarda el checkpoint ``fase_0_raw``.

        Args:
            filepath: Ruta al archivo de datos.
        """
        self.df = self.safe_execute(self._check_memory_and_load, filepath)

        self.checkpoint_manager.save_checkpoint(
            "fase_0_raw",
            self.df,
            metadata={
                "pipeline_id": self.pipeline_id,
                "phase": "fase_0_ingest",
                "rows": len(self.df),
                "columns": list(self.df.columns),
                "column_mapping": self.column_mapping,
            },
        )
        self.logger.info("Fase 0 - Ingesta completada y checkpoint guardado")

    def _create_masking_dict(self) -> Dict[str, str]:
        """Genera y aplica un mapeo de enmascaramiento de columnas.

        Crea nombres genéricos ``Var_XX`` para cada columna del
        ``DataFrame`` actual, guarda el mapeo en ``self.column_mapping`` y
        renombra ``self.df`` con esos nombres. Esto evita que nombres reales
        de columnas (posiblemente sensibles) viajen al exterior durante la
        extracción de metadatos.

        Returns:
            Diccionario ``{nombre_real: nombre_generico}``.

        Raises:
            RuntimeError: Si ``self.df`` aún no ha sido cargado.
        """
        if self.df is None:
            msg = "No hay datos cargados. Ejecuta load_data() primero."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.column_mapping = {
            col: f"Var_{idx:02d}" for idx, col in enumerate(self.df.columns, start=1)
        }
        self.df = self.df.rename(columns=self.column_mapping)

        self.logger.info(
            "Columnas enmascaradas: %d columnas renombradas",
            len(self.column_mapping),
        )
        return self.column_mapping

    def _infer_column_type(self, series: pd.Series) -> str:
        """Inferir el tipo semántico de una columna.

        Clasifica una columna como ``Numérico``, ``Categórico`` o
        ``Texto Libre``. El criterio para Texto Libre es que la columna sea
        de tipo objeto/string y su longitud media de cadenas sea mayor a 50.

        Args:
            series: Serie de Pandas a clasificar.

        Returns:
            Cadena con el tipo inferido.
        """
        if pd.api.types.is_numeric_dtype(series):
            return "Numérico"

        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            try:
                mean_length = series.astype(str).str.len().mean()
                if mean_length > 50:
                    return "Texto Libre"
            except Exception:
                pass
            return "Categórico"

        return "Categórico"

    def _compute_numeric_stats(self, series: pd.Series) -> Dict[str, Optional[float]]:
        """Calcula estadísticas descriptivas para una columna numérica.

        Los cálculos fallidos (p. ej. columna con un solo valor o todos
        nulos) se capturan silenciosamente y retornan ``None``.

        Args:
            series: Serie numérica.

        Returns:
            Diccionario con ``media``, ``std``, ``skewness`` y ``kurtosis``.
        """
        clean = series.dropna()
        result: Dict[str, Optional[float]] = {
            "media": None,
            "std": None,
            "skewness": None,
            "kurtosis": None,
        }

        if len(clean) == 0:
            return result

        def _safe_float(value: Any) -> Optional[float]:
            """Convierte un valor numérico a float, descartando NaN/Inf."""
            try:
                f = float(value)
                if pd.isna(f) or pd.isinf(f):
                    return None
                return f
            except Exception:
                return None

        result["media"] = _safe_float(clean.mean())
        result["std"] = _safe_float(clean.std())
        result["skewness"] = _safe_float(clean.skew())
        result["kurtosis"] = _safe_float(clean.kurtosis())

        return result

    def _compute_categorical_stats(
        self, series: pd.Series
    ) -> Dict[str, Optional[Union[str, int]]]:
        """Calcula estadísticas descriptivas para una columna categórica/texto.

        Args:
            series: Serie categórica o de texto.

        Returns:
            Diccionario con ``moda`` y ``frecuencia_moda``.
        """
        clean = series.dropna().astype(str)
        result: Dict[str, Optional[Union[str, int]]] = {
            "moda": None,
            "frecuencia_moda": None,
        }

        if len(clean) == 0:
            return result

        try:
            mode_value = clean.mode()
            if not mode_value.empty:
                result["moda"] = str(mode_value.iloc[0])
                result["frecuencia_moda"] = int((clean == result["moda"]).sum())
        except Exception:
            pass

        return result

    def _build_column_metadata(self, series: pd.Series) -> Dict[str, Any]:
        """Construye el metadato completo para una sola columna.

        Args:
            series: Serie de Pandas.

        Returns:
            Diccionario con tipo inferido, nulos, cardinalidad y estadísticas.
        """
        col_type = self._infer_column_type(series)
        total = len(series)
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total) * 100, 4) if total > 0 else 0.0
        cardinality = int(series.nunique(dropna=True))

        metadata: Dict[str, Any] = {
            "tipo": col_type,
            "total_valores": total,
            "valores_nulos": null_count,
            "pct_nulos": null_pct,
            "cardinalidad": cardinality,
        }

        if col_type == "Numérico":
            metadata["estadisticas"] = self._compute_numeric_stats(series)
        else:
            metadata["estadisticas"] = self._compute_categorical_stats(series)

        return metadata

    def _extract_metadata_core(self) -> Dict[str, Any]:
        """Núcleo de extracción de metadatos sobre ``self.df``.

        Aplica el enmascaramiento de columnas y genera la radiografía
        estadística completa del ``DataFrame``.

        Returns:
            Diccionario con metadatos globales y por columna.
        """
        if self.df is None:
            msg = "No hay datos cargados. Ejecuta load_data() primero."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self._create_masking_dict()

        columns_metadata = {
            col: self._build_column_metadata(self.df[col])
            for col in self.df.columns
        }

        metadata = {
            "pipeline_id": self.pipeline_id,
            "phase": "fase_0_metadata",
            "rows": len(self.df),
            "columns_count": len(self.df.columns),
            "columns": list(self.df.columns),
            "column_mapping": self.column_mapping,
            "columns_metadata": columns_metadata,
        }

        self.logger.info(
            "Metadatos extraídos: rows=%d | cols=%d",
            metadata["rows"],
            metadata["columns_count"],
        )
        return metadata

    def extract_metadata(self) -> Dict[str, Any]:
        """Extrae metadatos de forma segura y persiste el resultado.

        Envuelve la extracción con ``safe_execute`` y guarda un checkpoint
        ``fase_0_metadata`` con el diccionario de metadatos serializado.

        Returns:
            Diccionario completo de metadatos del dataset.
        """
        metadata = self.safe_execute(self._extract_metadata_core)

        self.checkpoint_manager.save_checkpoint(
            "fase_0_metadata",
            self.df,
            metadata=metadata,
        )
        self.logger.info("Checkpoint de metadatos guardado")
        return metadata

    def _resolve_target(self, raw_target: str) -> Optional[str]:
        """Traduce un nombre real de target a su nombre enmascarado.

        Limpia espacios y comillas del input del usuario. Si el usuario
        ingresó el nombre REAL de la columna (ej. 'Churn'), se traduce al
        nombre ENMASCARADO (ej. 'Var_21') usando ``column_mapping``. Si
        ingresó directamente el enmascarado, se deja igual.

        Args:
            raw_target: Nombre real de columna introducido por el usuario.

        Returns:
            Nombre enmascarado ``Var_XX`` o ``None`` si no se proporcionó
            entrada o la columna no existe en el mapeo.
        """
        target = raw_target.strip().strip('"').strip("'")
        if not target:
            return None

        if target in self.column_mapping:
            return self.column_mapping[target]

        # También acepta si el usuario ya escribió el nombre enmascarado.
        if target in self.column_mapping.values():
            return target

        self.logger.warning("Target '%s' no encontrado en el dataset", target)
        return None

    def determine_route(self, autonomous: bool = False) -> Tuple[str, str]:
        """Determina la ruta del pipeline según el target y los metadatos.

        Solicita al usuario el nombre de la variable objetivo, lo traduce al
        nombre enmascarado y aplica las reglas de enrutamiento:

        - ``fase_1``: existe target y no es texto libre continuo.
        - ``fase_3``: la mayoría de columnas (o el target) son texto libre.
        - ``fase_2``: cualquier otro caso (no supervisado estructurado).

        Returns:
            Tupla ``(ruta, target_enmascarado)``. El target será una cadena
            vacía si el usuario no definió uno.
        """
        if self.df is None:
            msg = "No hay datos cargados. Ejecuta load_data() primero."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if autonomous:
            # En modo autónomo inferimos que la última columna es el target
            raw_target = self.df.columns[-1]
            self.logger.info("Modo autónomo: asumiendo '%s' como variable objetivo", raw_target)
        else:
            raw_target = input(
                "Ingrese el nombre exacto de la variable objetivo (Target) "
                "o presione Enter si no existe: "
            )
        target = self._resolve_target(raw_target) or ""

        # Conteo de tipos sobre columnas enmascaradas.
        type_counts: Dict[str, int] = {"Numérico": 0, "Categórico": 0, "Texto Libre": 0}
        for col in self.df.columns:
            inferred = self._infer_column_type(self.df[col])
            type_counts[inferred] = type_counts.get(inferred, 0) + 1

        total_cols = len(self.df.columns)
        free_text_majority = (type_counts["Texto Libre"] / total_cols) > 0.5

        route = "fase_2"
        reason = "No hay target definido y predominio de variables estructuradas"

        if target:
            target_type = self._infer_column_type(self.df[target])
            if target_type == "Texto Libre":
                route = "fase_3"
                reason = "El target es texto libre continuo"
            else:
                route = "fase_1"
                reason = "Existe target estructurado"
        elif free_text_majority:
            route = "fase_3"
            reason = "Mayoría de columnas son texto libre"

        self.logger.info("Ruta determinada: %s | target: %s | motivo: %s", route, target, reason)
        return route, target

    def _drop_high_missing_columns(self, threshold: float) -> List[str]:
        """Elimina columnas cuyo porcentaje de nulos supere el umbral.

        Args:
            threshold: Umbral máximo de valores nulos (entre 0 y 1).

        Returns:
            Lista de columnas eliminadas.
        """
        if self.df is None:
            return []

        drop_cols = []
        for col in self.df.columns:
            null_pct = self.df[col].isna().mean()
            if null_pct > threshold:
                drop_cols.append(col)

        if drop_cols:
            self.df = self.df.drop(columns=drop_cols)
            self.logger.info(
                "Columnas eliminadas por exceso de nulos (>%.2f%%): %s",
                threshold * 100,
                drop_cols,
            )
        return drop_cols

    def _impute_missing_values(self) -> None:
        """Imputa valores nulos con estrategias deterministas.

        - Numéricos: mediana.
        - Categóricos/Texto: moda.
        """
        if self.df is None:
            return

        imputed_count = 0
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            if null_count == 0:
                continue

            col_type = self._infer_column_type(self.df[col])
            if col_type == "Numérico":
                median_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_value)
                self.logger.debug("Columna %s imputada con mediana %s", col, median_value)
            else:
                mode_value = self.df[col].mode()
                if not mode_value.empty:
                    fill_value = mode_value.iloc[0]
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.logger.debug("Columna %s imputada con moda %s", col, fill_value)
                else:
                    self.df[col] = self.df[col].fillna("MISSING")
                    self.logger.debug("Columna %s imputada con 'MISSING'", col)
            imputed_count += 1

        self.logger.info("Imputación determinista completada en %d columnas", imputed_count)

    def _apply_llm_recommendations(
        self,
        recommendations: List[ColumnRecommendation],
    ) -> None:
        """Aplica las recomendaciones de limpieza devueltas por el LLM.

        Args:
            recommendations: Lista de recomendaciones por columna.
        """
        if self.df is None:
            return

        drop_cols = []
        for rec in recommendations:
            col = rec.column_name
            if col not in self.df.columns:
                self.logger.warning("La columna %s no existe; se ignora recomendación", col)
                continue

            action = rec.action.lower().strip()
            if action == "drop":
                drop_cols.append(col)
            elif action == "impute_median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif action == "impute_mode":
                mode_value = self.df[col].mode()
                fill_value = mode_value.iloc[0] if not mode_value.empty else "MISSING"
                self.df[col] = self.df[col].fillna(fill_value)
            elif action in {"keep", "scale", "encode"}:
                self.logger.debug("Acción '%s' para %s marcada para fases posteriores", action, col)
            else:
                self.logger.warning("Acción '%s' no reconocida para %s", action, col)

        if drop_cols:
            self.df = self.df.drop(columns=drop_cols)
            self.logger.info("Columnas eliminadas por recomendación LLM: %s", drop_cols)

    def _query_llm_for_cleaning(self, metadata: Dict[str, Any]) -> MetadataResponse:
        """Consulta a Kimi (Moonshot API) para obtener recomendaciones de limpieza.

        Args:
            metadata: Diccionario de metadatos extraídos.

        Returns:
            Objeto ``MetadataResponse`` con las recomendaciones.

        Raises:
            RuntimeError: Si no está configurada la API key o falla la llamada.
        """
        if OpenAI is None:
            raise RuntimeError("La librería openai no está instalada")

        api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("La variable de entorno MOONSHOT_API_KEY no está configurada")

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
        )

        prompt = (
            "Eres un asistente de preprocesamiento de datos. "
            "A continuación recibirás un JSON con metadatos de un dataset "
            "cuyas columnas están enmascaradas como Var_XX. "
            "Devuelve un JSON estrictamente válido que represente una estrategia general y una lista de recomendaciones "
            "por columna. Las acciones permitidas son: "
            "'impute_median', 'impute_mode', 'drop', 'scale', 'encode', 'keep'. "
            "También sugiere la ruta más adecuada: 'fase_1', 'fase_2' o 'fase_3'.\n\n"
            "Formato esperado del JSON:\n"
            "{\n"
            '  "overall_strategy": "tu estrategia",\n'
            '  "column_recommendations": [{"column_name": "Var_01", "action": "keep", "reason": "motivo"}],\n'
            '  "suggested_route": "fase_1"\n'
            "}\n\n"
            f"Metadatos:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}"
        )

        try:
            response = client.chat.completions.create(
                model="kimi-k2.6",
                messages=[
                    {"role": "system", "content": "Devuelve únicamente JSON. No incluyas backticks de markdown (```json)."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return MetadataResponse(**parsed)
        except Exception as exc:
            self.logger.error("Error al consultar Kimi: %s", exc)
            raise RuntimeError(f"Fallo la consulta al LLM: {exc}") from exc

    def orchestrate_cleaning(self, ai_mode: bool = False) -> None:
        """Orquesta la limpieza estructural de forma determinista o con LLM.

        Args:
            ai_mode: Si es ``True``, usa Kimi para recomendar pasos de
                limpieza. Si es ``False`` (por defecto), aplica reglas
                deterministas de privacidad máxima.

        Raises:
            RuntimeError: Si ``self.df`` no está cargado o falla el modo AI.
        """
        if self.df is None:
            msg = "No hay datos cargados. Ejecuta load_data() primero."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if ai_mode:
            self.logger.info("Orquestador en modo AI: consultando Moonshot (Kimi)")
            metadata = self._extract_metadata_core()
            llm_response = self.safe_execute(self._query_llm_for_cleaning, metadata)
            self._apply_llm_recommendations(llm_response.column_recommendations)
            self.logger.info("Limpieza con LLM completada")
        else:
            self.logger.info("Orquestador en modo determinista (máxima privacidad)")
            max_missing_pct = self.config.get_param(
                "statistics.max_missing_pct",
                default=0.60,
            )
            self._drop_high_missing_columns(max_missing_pct)
            self._impute_missing_values()
            self.logger.info("Limpieza determinista completada")

    def execute(
        self,
        filepath: str,
        ai_mode: bool = False,
        autonomous: bool = False,
    ) -> Tuple[str, str]:
        """Ejecuta el flujo completo de la Fase 0.

        Orquesta: carga, extracción de metadatos, limpieza, determinación de
        ruta y checkpoint final.

        Args:
            filepath: Ruta al archivo de datos.
            ai_mode: Si es ``True``, activa el modo de limpieza con LLM.
            autonomous: Si es ``True``, infiere valores sin pedir input al usuario.

        Returns:
            Tupla ``(ruta_sugerida, target_enmascarado)``.
        """
        self.logger.info("Iniciando Fase 0 - Ejecución completa")

        self.load_data(filepath)
        self.extract_metadata()
        self.orchestrate_cleaning(ai_mode=ai_mode)
        route, target = self.determine_route(autonomous=autonomous)

        self.checkpoint_manager.save_checkpoint(
            "fase_0_clean",
            self.df,
            metadata={
                "pipeline_id": self.pipeline_id,
                "phase": "fase_0_clean",
                "route": route,
                "target": target,
                "ai_mode": ai_mode,
                "rows": len(self.df),
                "columns": list(self.df.columns),
                "column_mapping": self.column_mapping,
            },
        )
        self.logger.info(
            "Fase 0 - Ejecución completa finalizada | ruta=%s | target=%s | ai_mode=%s",
            route,
            target,
            ai_mode,
        )
        return route, target

    def run(self, filepath: str) -> Dict[str, Any]:
        """Ejecuta la fase completa de ingesta y extracción de metadatos.

        Args:
            filepath: Ruta al archivo de datos.

        Returns:
            Diccionario completo de metadatos extraídos.
        """
        self.logger.info("Iniciando Fase 0 - Ingesta")
        self.load_data(filepath)
        metadata = self.extract_metadata()
        self.logger.info("Fase 0 - Ingesta finalizada")
        return metadata
