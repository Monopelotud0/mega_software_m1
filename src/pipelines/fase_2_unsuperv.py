"""Fase 2: El Camino No Supervisado (Clustering).

Esta fase recibe el DataFrame limpio producido por la Fase 0 y realiza:

- Detección y separación de anomalías mediante ``IsolationForest``.
- Construcción de un espacio geométrico listo para algoritmos de clustering,
  aplicando escalado robusto a variables numéricas y codificación one-hot a
  variables categóricas.

El resultado es un dataset limpio de outliers y una matriz transformada que
puede ser consumida por técnicas de clustering (por ejemplo HDBSCAN o K-Means)
en pasos posteriores.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.pipelines.base_pipeline import BasePipeline


class UnsupervisedPipeline(BasePipeline):
    """Pipeline para tareas no supervisadas (clustering y detección de anomalías).

    Hereda de ``BasePipeline`` y carga el checkpoint ``fase_0_clean`` para
    partir desde un dataset ya limpio y enmascarado.

    Attributes:
        df_clean: DataFrame sin las anomalías detectadas (inliers).
        df_anomalies: DataFrame con las anomalías etiquetadas (outliers).
        preprocessor: Pipeline de sklearn con transformación numérica y
            categórica listo para producir la matriz geométrica.
    """

    def __init__(
        self,
        pipeline_id: str,
        params_path: Optional[str] = None,
    ) -> None:
        """Inicializa la fase no supervisada."""
        super().__init__(pipeline_id, params_path)
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_anomalies: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.pca: Optional[PCA] = None
        self.best_kmeans: Optional[MiniBatchKMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None

        self.logger.info("Fase 2 - No Supervisado inicializada")

    def _remove_outliers(self) -> None:
        """Detecta y separa outliers usando IsolationForest.

        Carga el checkpoint ``fase_0_clean``, selecciona las columnas
        numéricas y entrena un ``IsolationForest`` para etiquetar inliers
        (+1) y outliers (-1). El DataFrame original se divide en
        ``self.df_clean`` y ``self.df_anomalies``.

        Raises:
            RuntimeError: Si no existe el checkpoint ``fase_0_clean``.
            ValueError: Si el DataFrame cargado está vacío.
        """
        try:
            df, metadata = self.checkpoint_manager.load_checkpoint("fase_0_clean")
            if df is None:
                raise RuntimeError("No existe el checkpoint 'fase_0_clean'")

            self.df = df.copy()
            self.logger.info(
                "Datos cargados para detección de anomalías: rows=%d | cols=%d",
                len(self.df),
                len(self.df.columns),
            )

            if self.df.empty:
                raise ValueError("El DataFrame cargado está vacío")

            numeric_cols: List[str] = self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            if not numeric_cols:
                self.logger.warning(
                    "No se encontraron columnas numéricas; se omite la detección de anomalías"
                )
                self.df_clean = self.df.copy()
                self.df_anomalies = pd.DataFrame(columns=self.df.columns)
                return

            iso = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
            )
            labels = iso.fit_predict(self.df[numeric_cols])

            self.df["__anomaly_label__"] = labels
            n_anomalies = int((labels == -1).sum())
            self.logger.info("Anomalías detectadas: %d de %d filas", n_anomalies, len(self.df))

            self.df_clean = self.df[self.df["__anomaly_label__"] == 1].drop(
                columns=["__anomaly_label__"]
            )
            self.df_anomalies = self.df[self.df["__anomaly_label__"] == -1].drop(
                columns=["__anomaly_label__"]
            )
            self.df_anomalies["__anomaly__"] = -1

            self.logger.info(
                "Dataset limpio: rows=%d | Anomalías: rows=%d",
                len(self.df_clean),
                len(self.df_anomalies),
            )
        except Exception:
            self.logger.exception("Error en _remove_outliers")
            raise

    def _build_geometric_space(self) -> np.ndarray:
        """Construye la matriz geométrica para clustering.

        Crea un ``ColumnTransformer`` que aplica ``RobustScaler`` a las
        variables numéricas y ``OneHotEncoder`` a las categóricas. Entrena
        el transformador sobre ``self.df_clean`` y retorna la matriz
        resultante.

        Returns:
            Matriz numpy lista para alimentar algoritmos de clustering.

        Raises:
            RuntimeError: Si ``self.df_clean`` no ha sido definido.
            ValueError: Si no hay columnas numéricas ni categóricas para
                transformar.
        """
        try:
            if self.df_clean is None:
                raise RuntimeError("Debe ejecutarse _remove_outliers antes de construir el espacio geométrico")

            numeric_cols: List[str] = self.df_clean.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            categorical_cols: List[str] = self.df_clean.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()

            self.logger.info(
                "Columnas numéricas=%d | categóricas=%d",
                len(numeric_cols),
                len(categorical_cols),
            )

            transformers: List[Any] = []
            if numeric_cols:
                transformers.append(("num", RobustScaler(), numeric_cols))
            if categorical_cols:
                transformers.append(
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_cols,
                    )
                )

            if not transformers:
                raise ValueError("No hay columnas numéricas ni categóricas para transformar")

            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
            )

            geometric_space = self.preprocessor.fit_transform(self.df_clean)
            self.logger.info(
                "Espacio geométrico construido: shape=%s",
                geometric_space.shape,
            )
            return geometric_space
        except Exception:
            self.logger.exception("Error en _build_geometric_space")
            raise

    def _reduce_dimensionality(self, geometric_matrix: np.ndarray) -> np.ndarray:
        """Reduce la dimensionalidad con PCA y detecta ausencia de estructura.

        Ajusta un ``PCA()`` sobre la matriz geométrica, calcula la varianza
        explicada acumulada y selecciona el número mínimo de componentes que
        alcanzan al menos el 80% de varianza. Si dicho número supera el 70%
        de las columnas originales, emite una alerta de posible ausencia de
        estructura latente.

        Args:
            geometric_matrix: Matriz resultante de ``_build_geometric_space``.

        Returns:
            Matriz reducida al número óptimo de componentes principales.

        Raises:
            ValueError: Si la matriz de entrada está vacía.
        """
        try:
            if geometric_matrix.size == 0:
                raise ValueError("La matriz geométrica está vacía")

            original_dims = geometric_matrix.shape[1]
            self.pca = PCA()
            self.pca.fit(geometric_matrix)

            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = int(np.argmax(cumulative_variance >= 0.80) + 1)

            threshold = int(np.ceil(0.70 * original_dims))
            if n_components > threshold:
                self.logger.warning(
                    "Posible ausencia de estructura latente (datos ruidosos): "
                    "componentes necesarios=%d (%.1f%% de %d dimensiones originales)",
                    n_components,
                    100.0 * n_components / original_dims,
                    original_dims,
                )

            self.logger.info(
                "PCA: componentes óptimos=%d | varianza explicada=%.4f",
                n_components,
                cumulative_variance[n_components - 1],
            )

            self.pca = PCA(n_components=n_components)
            reduced_matrix = self.pca.fit_transform(geometric_matrix)
            self.logger.info("Matriz reducida: shape=%s", reduced_matrix.shape)
            return reduced_matrix
        except Exception:
            self.logger.exception("Error en _reduce_dimensionality")
            raise

    def _find_optimal_k(
        self, reduced_matrix: np.ndarray
    ) -> tuple[int, MiniBatchKMeans, np.ndarray]:
        """Descubre empíricamente el número óptimo de clusters.

        Evalúa valores de ``k`` entre 2 y ``min(10, n_filas // 10)`` usando
        ``MiniBatchKMeans``. Selecciona el ``k`` que mejor equilibre un
        Silhouette Score alto y un Davies-Bouldin Score bajo.

        Args:
            reduced_matrix: Matriz reducida por PCA.

        Returns:
            Tupla con el mejor ``k``, el modelo ``MiniBatchKMeans`` ganador y
            las etiquetas de cluster asignadas.

        Raises:
            ValueError: Si no hay suficientes filas para clustering.
        """
        try:
            n_rows = reduced_matrix.shape[0]
            if n_rows < 4:
                raise ValueError("Se necesitan al menos 4 filas para evaluar clusters")

            max_k = min(10, n_rows // 10)
            if max_k < 2:
                max_k = 2

            best_k: int = 2
            best_score: float = -np.inf
            self.best_kmeans = None
            self.cluster_labels = None

            for k in range(2, max_k + 1):
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(reduced_matrix)

                sil = silhouette_score(reduced_matrix, labels)
                db = davies_bouldin_score(reduced_matrix, labels)

                # Normalizar ambas métricas al rango [0, 1] para combinarlas.
                # Silhouette ya está en [-1, 1]; lo llevamos a [0, 1].
                sil_norm = (sil + 1.0) / 2.0
                # Davies-Bouldin >= 0; usar transformación inversa suavizada.
                db_norm = 1.0 / (1.0 + db)

                combined_score = 0.6 * sil_norm + 0.4 * db_norm

                self.logger.info(
                    "k=%d | Silhouette=%.4f | Davies-Bouldin=%.4f | combined=%.4f",
                    k,
                    sil,
                    db,
                    combined_score,
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    self.best_kmeans = kmeans
                    self.cluster_labels = labels

            self.logger.info(
                "K óptimo seleccionado: k=%d | combined_score=%.4f",
                best_k,
                best_score,
            )
            return best_k, self.best_kmeans, self.cluster_labels
        except Exception:
            self.logger.exception("Error en _find_optimal_k")
            raise

    def _validate_clusters(self, labels: np.ndarray) -> List[str]:
        """Valida estadísticamente qué variables diferencian los clusters.

        Añade ``labels`` como columna ``Cluster_ID`` a ``self.df_clean`` y
        evalúa, para cada columna original, si existe una asociación
        estadísticamente significativa con el cluster:

        - Variables numéricas: ANOVA ``f_oneway``.
        - Variables categóricas: Chi-cuadrado ``chi2_contingency``.

        Args:
            labels: Etiquetas de cluster asignadas a cada fila de
                ``self.df_clean``.

        Returns:
            Lista con los nombres de las variables diferenciadoras clave
            (``p < 0.05``).

        Raises:
            RuntimeError: Si ``self.df_clean`` no ha sido definido.
        """
        try:
            if self.df_clean is None:
                raise RuntimeError("self.df_clean no está definido")

            df_eval = self.df_clean.copy()
            df_eval["Cluster_ID"] = labels

            numeric_cols = df_eval.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_eval.select_dtypes(exclude=[np.number]).columns.tolist()
            # Excluir la columna de cluster de las evaluaciones.
            if "Cluster_ID" in numeric_cols:
                numeric_cols.remove("Cluster_ID")
            if "Cluster_ID" in categorical_cols:
                categorical_cols.remove("Cluster_ID")

            key_differentiators: List[str] = []

            for col in numeric_cols:
                groups = [group[col].dropna().values for _, group in df_eval.groupby("Cluster_ID")]
                if len(groups) < 2 or any(len(g) == 0 for g in groups):
                    continue
                _, p_value = stats.f_oneway(*groups)
                if p_value < 0.05:
                    key_differentiators.append(col)
                    self.logger.debug("Diferenciador clave (numérico): %s (p=%.4f)", col, p_value)

            for col in categorical_cols:
                contingency = pd.crosstab(df_eval[col], df_eval["Cluster_ID"])
                if contingency.empty:
                    continue
                _, p_value, _, _ = stats.chi2_contingency(contingency)
                if p_value < 0.05:
                    key_differentiators.append(col)
                    self.logger.debug("Diferenciador clave (categórico): %s (p=%.4f)", col, p_value)

            self.logger.info(
                "Validación de clusters completada: %d diferenciadores clave encontrados",
                len(key_differentiators),
            )
            return key_differentiators
        except Exception:
            self.logger.exception("Error en _validate_clusters")
            raise

    def _generate_profiles(self, key_differentiators: List[str]) -> Dict[str, Any]:
        """Genera perfiles estadísticos por cluster.

        Agrupa ``self.df_clean`` por ``Cluster_ID`` y resume las variables
        diferenciadoras clave usando la mediana para numéricas y la moda
        para categóricas.

        Args:
            key_differentiators: Lista de variables diferenciadoras clave.

        Returns:
            Diccionario con perfiles por cluster.

        Raises:
            RuntimeError: Si ``self.df_clean`` no contiene ``Cluster_ID``.
            ValueError: Si la lista de diferenciadores está vacía.
        """
        try:
            if self.df_clean is None:
                raise RuntimeError("self.df_clean no está definido")
            if "Cluster_ID" not in self.df_clean.columns:
                raise RuntimeError("self.df_clean no contiene la columna Cluster_ID")
            if not key_differentiators:
                raise ValueError("No hay diferenciadores clave para perfilar")

            profiles: Dict[str, Any] = {}
            grouped = self.df_clean.groupby("Cluster_ID")

            for cluster_id, group in grouped:
                cluster_profile: Dict[str, Any] = {"size": int(len(group))}
                for col in key_differentiators:
                    if col in group.columns:
                        if pd.api.types.is_numeric_dtype(group[col]):
                            cluster_profile[col] = {
                                "median": float(group[col].median()),
                                "mean": float(group[col].mean()),
                                "std": float(group[col].std()),
                            }
                        else:
                            mode_value = group[col].mode()
                            cluster_profile[col] = {
                                "mode": mode_value.iloc[0] if not mode_value.empty else None,
                                "categories": group[col].value_counts().to_dict(),
                            }
                profiles[str(cluster_id)] = cluster_profile

            self.logger.info("Perfiles generados para %d clusters", len(profiles))
            return profiles
        except Exception:
            self.logger.exception("Error en _generate_profiles")
            raise

    def _save_model_artifact(self) -> Path:
        """Serializa el pipeline completo de transformación y clustering.

        Guarda un diccionario con el preprocessor, PCA y modelo KMeans en
        ``data/03_output/unsuperv_model.pkl``.

        Returns:
            Ruta del archivo serializado.
        """
        try:
            output_dir = Path("data") / "03_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "unsuperv_model.pkl"

            artifact = {
                "preprocessor": self.preprocessor,
                "pca": self.pca,
                "kmeans": self.best_kmeans,
            }
            joblib.dump(artifact, model_path)
            self.logger.info("Modelo no supervisado serializado: %s", model_path.resolve())
            return model_path
        except Exception:
            self.logger.exception("Error en _save_model_artifact")
            raise

    def _generate_visualizations(
        self, reduced_matrix: np.ndarray, labels: np.ndarray
    ) -> None:
        """Genera y guarda visualizaciones de la segmentación por clusters.

        Crea un scatter plot de las dos primeras componentes principales
        coloreado por etiqueta de cluster y lo persiste en
        ``data/03_output/figures/pca_clusters.png``.

        Args:
            reduced_matrix: Matriz reducida por PCA.
            labels: Etiquetas de cluster asignadas a cada fila.
        """
        try:
            figures_dir = Path("data") / "03_output" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=reduced_matrix[:, 0],
                y=reduced_matrix[:, 1],
                hue=labels,
                palette="tab10",
                legend="full",
                s=60,
                alpha=0.7,
            )
            plt.title("Segmentación espacial: PCA1 vs PCA2")
            plt.xlabel("PCA1")
            plt.ylabel("PCA2")
            plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            output_path = figures_dir / "pca_clusters.png"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            self.logger.info("Visualización de clusters guardada: %s", output_path.resolve())
        except Exception:
            self.logger.exception("Error en _generate_visualizations")
            raise

    def _generate_markdown_report(
        self,
        optimal_k: int,
        silhouette: float,
        davies: float,
        profiles: Dict[str, Any],
        total_anomalies: int,
    ) -> None:
        """Genera el reporte Markdown de la fase no supervisada.

        Persiste un resumen profesional de la segmentación en
        ``data/03_output/reporte_clustering.md``.

        Args:
            optimal_k: Número óptimo de clusters descubierto.
            silhouette: Coeficiente de silueta del modelo ganador.
            davies: Índice Davies-Bouldin del modelo ganador.
            profiles: Diccionario con perfiles por cluster.
            total_anomalies: Cantidad de anomalías detectadas y aisladas.
        """
        try:
            output_dir = Path("data") / "03_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "reporte_clustering.md"

            report_lines: List[str] = [
                "# Reporte de Segmentación (Clustering)",
                "",
                f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 1. Limpieza Espacial",
                "",
                f"- Anomalías detectadas y aisladas: **{total_anomalies}**",
                "",
                "## 2. Búsqueda del Óptimo Matemático",
                "",
                f"- Clústers descubiertos (K): **{optimal_k}**",
                f"- Coeficiente de Silueta: **{silhouette:.4f}**",
                f"- Índice Davies-Bouldin: **{davies:.4f}**",
                "",
                "## 3. Visualización Espacial (PCA)",
                "",
                "![Segmentación PCA](figures/pca_clusters.png)",
                "",
                "## 4. Perfilado de Clústers (Diferenciadores Claves)",
                "",
                "```json",
                str(profiles),
                "```",
                "",
            ]

            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            self.logger.info("Reporte de clustering guardado: %s", report_path.resolve())
        except Exception:
            self.logger.exception("Error en _generate_markdown_report")
            raise

    def execute(self) -> None:
        """Ejecuta la Fase 2 completa de forma encadenada.

        Flujo: detección de anomalías -> espacio geométrico -> reducción de
        dimensionalidad -> descubrimiento de clusters -> validación estadística
        -> generación de perfiles -> visualizaciones -> reporte Markdown ->
        guardado de checkpoints y modelo.
        """
        try:
            self.safe_execute(self._remove_outliers)
            geometric_space = self.safe_execute(self._build_geometric_space)
            reduced_matrix = self.safe_execute(self._reduce_dimensionality, geometric_space)
            best_k, best_kmeans, labels = self.safe_execute(self._find_optimal_k, reduced_matrix)

            if self.df_clean is not None and labels is not None:
                self.df_clean["Cluster_ID"] = labels
            else:
                raise RuntimeError("No se pudieron asignar etiquetas de cluster")

            key_differentiators = self.safe_execute(self._validate_clusters, labels)
            profiles = self.safe_execute(self._generate_profiles, key_differentiators)

            silhouette = silhouette_score(reduced_matrix, labels)
            davies = davies_bouldin_score(reduced_matrix, labels)
            total_anomalies = (
                len(self.df_anomalies) if self.df_anomalies is not None else 0
            )

            self.safe_execute(
                self._generate_visualizations, reduced_matrix, labels
            )
            self.safe_execute(
                self._generate_markdown_report,
                best_k,
                silhouette,
                davies,
                profiles,
                total_anomalies,
            )

            # Reintegrar anomalías con Cluster_ID = -1.
            if self.df_anomalies is not None and not self.df_anomalies.empty:
                anomalies = self.df_anomalies.copy()
                anomalies["Cluster_ID"] = -1
                final_df = pd.concat([self.df_clean, anomalies], ignore_index=True)
            else:
                final_df = self.df_clean.copy()

            self.checkpoint_manager.save_checkpoint("fase_2_final", final_df)
            self.checkpoint_manager.save_checkpoint(
                "fase_2_profiles",
                pd.DataFrame([profiles]),
                metadata=profiles,
            )

            self.safe_execute(self._save_model_artifact)

            self.logger.info(
                "Fase 2 completada: k=%d | diferenciadores=%d | filas_finales=%d",
                best_k,
                len(key_differentiators),
                len(final_df),
            )
        except Exception:
            self.logger.exception("Error en execute de Fase 2")
            raise

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Punto de entrada principal de la fase no supervisada.

        Delegación transparente a ``execute``.
        """
        self.execute()
