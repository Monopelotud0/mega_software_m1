"""Fase 1: Camino Supervisado del Mega Software de ML.

Esta fase recibe el DataFrame limpio producido por la Fase 0 y realiza:

- División entrenamiento/prueba con bloqueo de datos de test.
- Pruebas de hipótesis para seleccionar variables relevantes frente al target.
- Filtrado por Factor de Inflación de Varianza (VIF) para reducir
  multicolinealidad.

Soporta tanto tareas de clasificación como de regresión.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import shap
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.pipelines.base_pipeline import BasePipeline

#: Directorio de salida para modelos serializados.
OUTPUT_DIR = Path("data") / "03_output"


class SupervisedPipeline(BasePipeline):
    """Pipeline para tareas supervisadas (clasificación y regresión).

    Hereda de ``BasePipeline`` y carga el checkpoint ``fase_0_clean`` para
    partir desde un dataset ya limpio y enmascarado.

    Attributes:
        X_train: Características de entrenamiento.
        X_test: Características de prueba (bloqueadas para análisis).
        y_train: Target de entrenamiento.
        y_test: Target de prueba (bloqueado para análisis).
        task_type: Tipo de tarea: ``'classification'`` o ``'regression'``.
        p_values: Diccionario ``{columna: p_value}`` del test de hipótesis.
        original_columns: Lista de columnas originales al cargar el checkpoint.
    """

    def __init__(
        self,
        pipeline_id: str,
        params_path: Optional[str] = None,
    ) -> None:
        """Inicializa la fase supervisada."""
        super().__init__(pipeline_id, params_path)
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.task_type: Optional[str] = None
        self.p_values: Dict[str, float] = {}
        self.original_columns: List[str] = []
        self.model: Optional[Pipeline] = None
        self.cv_scores: Dict[str, List[float]] = {}
        self.selected_model_name: Optional[str] = None
        self.stats_report: Dict[str, Any] = {}
        self.column_mapping_inv: Dict[str, str] = {}

        self.logger.info("Fase 1 - Supervisado inicializada")

    def _load_clean_data(self) -> pd.DataFrame:
        """Carga el checkpoint ``fase_0_clean``.

        Returns:
            DataFrame limpio producido por la Fase 0.

        Raises:
            RuntimeError: Si no existe el checkpoint.
        """
        df, metadata = self.checkpoint_manager.load_checkpoint("fase_0_clean")
        if df is None:
            msg = "No se encontró el checkpoint 'fase_0_clean'. Ejecuta la Fase 0 primero."
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        self.column_mapping_inv = {v: k for k, v in metadata.get("column_mapping", {}).items()}

        self.original_columns = list(df.columns)
        self.logger.info(
            "Checkpoint fase_0_clean cargado: rows=%d | cols=%d",
            len(df),
            len(df.columns),
        )
        return df

    def _data_splitting(
        self,
        target_col: str,
        task_type: str,
    ) -> None:
        """Divide el dataset en entrenamiento y prueba.

        Los datos de test quedan bloqueados: solo se usan ``X_train`` e
        ``y_train`` en los análisis posteriores.

        Args:
            target_col: Nombre de la columna objetivo (puede ser real o
                enmascarado ``Var_XX``).
            task_type: ``'classification'`` o ``'regression'``.

        Raises:
            ValueError: Si el target no existe o el tipo de tarea es
                inválido.
        """
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type debe ser 'classification' o 'regression'")

        df = self._load_clean_data()

        if target_col not in df.columns:
            msg = f"La columna target '{target_col}' no existe en el dataset"
            self.logger.error(msg)
            raise ValueError(msg)

        self.task_type = task_type
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if task_type == "classification":
            self._label_encoder = LabelEncoder()
            y = pd.Series(
                self._label_encoder.fit_transform(y),
                index=y.index,
                name=y.name,
            )
            self.logger.info(
                "Target codificado con LabelEncoder. Clases originales: %s",
                list(self._label_encoder.classes_),
            )

        split_kwargs: Dict[str, Any] = {
            "test_size": 0.20,
            "random_state": 42,
        }
        if task_type == "classification":
            split_kwargs["stratify"] = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, **split_kwargs
        )

        self.logger.info(
            "División completada: train=%d | test=%d | task=%s",
            len(self.X_train),
            len(self.X_test),
            task_type,
        )
        self.logger.warning(
            "Datos de test bloqueados. Solo se utilizarán X_train e y_train "
            "en los análisis posteriores."
        )

    def _is_numeric(self, series: pd.Series) -> bool:
        """Determina si una serie es numérica continua."""
        return pd.api.types.is_numeric_dtype(series)

    def _hypothesis_testing(
        self,
        target_col: str,
        task_type: str,
    ) -> List[str]:
        """Selecciona columnas relevantes mediante pruebas de hipótesis.

        Evalúa cada columna de ``X_train`` contra ``y_train`` y retorna las
        columnas cuyo p-value sea menor al umbral configurado.

        Args:
            target_col: Nombre de la columna objetivo.
            task_type: ``'classification'`` o ``'regression'``.

        Returns:
            Lista de columnas aprobadas.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Los datos no han sido divididos. Ejecuta _data_splitting primero.")

        threshold = float(
            self.config.get_param("statistics.p_value_threshold", default=0.05)
        )
        approved_cols: List[str] = []
        self.p_values = {}

        for col in self.X_train.columns:
            x_col = self.X_train[col]
            p_value: Optional[float] = None

            try:
                if self._is_numeric(x_col):
                    if task_type == "classification":
                        # ANOVA: grupos definidos por y_train
                        groups = [x_col[self.y_train == cls].dropna() for cls in self.y_train.unique()]
                        if all(len(g) > 0 for g in groups):
                            _, p_value = stats.f_oneway(*groups)
                    else:  # regression
                        mask = x_col.notna() & self.y_train.notna()
                        if mask.sum() > 2:
                            _, p_value = stats.pearsonr(x_col[mask], self.y_train[mask])
                else:
                    if task_type == "classification":
                        contingency = pd.crosstab(x_col, self.y_train)
                        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                            _, p_value, _, _ = stats.chi2_contingency(contingency)
                    else:
                        # Para regresión con categórica: ANOVA de y ~ grupos
                        groups = [
                            self.y_train[x_col == cat].dropna()
                            for cat in x_col.unique()
                        ]
                        if all(len(g) > 0 for g in groups):
                            _, p_value = stats.f_oneway(*groups)
            except Exception as exc:
                self.logger.debug("No se pudo calcular p-value para %s: %s", col, exc)
                p_value = None

            self.p_values[col] = p_value if p_value is not None else 1.0

            if p_value is not None and p_value < threshold:
                approved_cols.append(col)
                self.logger.debug("Columna aprobada: %s (p=%.4f)", col, p_value)
            else:
                self.logger.debug("Columna rechazada: %s (p=%.4f)", col, p_value)

        rejected_cols = [c for c in self.p_values if c not in approved_cols]
        self.stats_report["hypothesis"] = {
            "threshold": threshold,
            "total": len(self.p_values),
            "approved_count": len(approved_cols),
            "rejected_count": len(rejected_cols),
            "approved_columns": approved_cols,
            "rejected_columns": rejected_cols,
        }

        self.X_train = self.X_train[approved_cols]
        self.logger.info(
            "Pruebas de hipótesis completadas: %d/%d columnas aprobadas (threshold=%.4f)",
            len(approved_cols),
            len(self.p_values),
            threshold,
        )
        return approved_cols

    def _vif_filter(self, num_cols: List[str]) -> List[str]:
        """Reduce multicolinealidad mediante poda iterativa de VIF.

        Calcula el VIF para las columnas numéricas aprobadas y elimina
        iterativamente la variable con mayor VIF mientras supere el umbral.
        En caso de empate, se desempata por mayor p-value.

        Args:
            num_cols: Lista de columnas numéricas aprobadas.

        Returns:
            Lista de columnas numéricas que sobreviven al filtro VIF.
        """
        if self.X_train is None:
            raise RuntimeError("Los datos no han sido divididos.")

        if not num_cols:
            return []

        vif_threshold = float(
            self.config.get_param("statistics.vif_threshold", default=10.0)
        )

        # Filtrar solo columnas numéricas presentes en X_train actual
        remaining_cols = [c for c in num_cols if c in self.X_train.columns]
        if not remaining_cols:
            return []

        while True:
            df_numeric = self.X_train[remaining_cols].select_dtypes(include=[np.number]).dropna()
            if df_numeric.empty or len(df_numeric.columns) < 2:
                break

            X_const = sm.add_constant(df_numeric, has_constant="add")
            vif_data = pd.DataFrame()
            vif_data["feature"] = df_numeric.columns
            vif_data["vif"] = [
                variance_inflation_factor(X_const.values, idx + 1)
                for idx in range(len(df_numeric.columns))
            ]
            vif_data["p_value"] = vif_data["feature"].map(lambda c: self.p_values.get(c, 1.0))

            max_vif = vif_data["vif"].max()
            if max_vif <= vif_threshold:
                break

            # Desempate por mayor p-value, luego mayor VIF
            candidate = vif_data.sort_values(
                by=["p_value", "vif"],
                ascending=[False, False],
            ).iloc[0]
            removed_col = str(candidate["feature"])
            remaining_cols.remove(removed_col)

            self.logger.info(
                "VIF alta (%.2f) en %s; columna eliminada",
                float(candidate["vif"]),
                removed_col,
            )

            if not remaining_cols:
                break

        removed_by_vif = [c for c in num_cols if c not in remaining_cols]
        self.stats_report["vif"] = {
            "threshold": vif_threshold,
            "original_numeric_count": len(num_cols),
            "surviving_numeric_count": len(remaining_cols),
            "removed_numeric_count": len(removed_by_vif),
            "removed_columns": removed_by_vif,
            "surviving_columns": remaining_cols,
        }

        self.logger.info(
            "Filtro VIF completado: %d/%d columnas numéricas sobreviven",
            len(remaining_cols),
            len(num_cols),
        )

        total_original = len(self.original_columns)
        remaining_total = len(remaining_cols) + sum(
            1 for c in self.X_train.columns if c not in remaining_cols
        )
        if total_original > 0:
            elimination_pct = (total_original - remaining_total) / total_original
            if elimination_pct > 0.50:
                self.logger.critical(
                    "Alerta crítica: se eliminó %.1f%% de las columnas originales "
                    "durante la selección y filtrado VIF",
                    elimination_pct * 100,
                )

        return remaining_cols

    def _build_preprocessor(
        self,
        num_cols: List[str],
        cat_cols: List[str],
    ) -> ColumnTransformer:
        """Construye el preprocesador de Scikit-Learn.

        Crea pipelines específicos para variables numéricas y categóricas.
        Si en el futuro se detecta desbalanceo de clases, aquí se puede
        inyectar SMOTE mediante ``imblearn.pipeline.Pipeline``.

        Args:
            num_cols: Columnas numéricas.
            cat_cols: Columnas categóricas.

        Returns:
            ``ColumnTransformer`` listo para usar en un pipeline.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        transformers = []
        if num_cols:
            transformers.append(("num", numeric_pipeline, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_pipeline, cat_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.logger.info(
            "Preprocesador construido: num=%d | cat=%d",
            len(num_cols),
            len(cat_cols),
        )
        return preprocessor

    def _get_cv_splitter(self, n_splits: int = 5):
        """Devuelve el divisor de validación cruzada adecuado a la tarea."""
        if self.task_type == "classification":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _get_scorer(self):
        """Devuelve el scorer adecuado a la tarea."""
        if self.task_type == "classification":
            return make_scorer(f1_score, average="weighted")
        return make_scorer(mean_squared_error, greater_is_better=False)

    def _get_cv_scoring_string(self) -> str:
        """Devuelve el nombre del scoring para ``cross_val_score``."""
        if self.task_type == "classification":
            return "f1_weighted"
        return "neg_root_mean_squared_error"

    def _define_models(self) -> Dict[str, Any]:
        """Define los modelos candidatos según el tipo de tarea.

        Returns:
            Diccionario ``{nombre: estimador}``.
        """
        if self.task_type == "classification":
            return {
                "baseline": DummyClassifier(strategy="most_frequent"),
                "linear": LogisticRegression(max_iter=1000, random_state=42),
                "forest": RandomForestClassifier(random_state=42, n_estimators=100),
                "gradient_boosting": HistGradientBoostingClassifier(random_state=42),
            }
        return {
            "baseline": DummyRegressor(strategy="median"),
            "linear": LinearRegression(),
            "forest": RandomForestRegressor(random_state=42, n_estimators=100),
            "gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        }

    def _train_and_select_model(
        self,
        preprocessor: ColumnTransformer,
    ) -> Pipeline:
        """Entrena y selecciona el modelo ganador por parsimonia.

        Realiza validación cruzada sobre los modelos candidatos. Compara el
        modelo más complejo (``gradient_boosting``) contra el más simple
        explicable (``linear``) mediante una prueba T pareada. Si no hay
        diferencia estadísticamente significativa (p-value > 0.05), gana el
        modelo simple por principio de explicabilidad.

        Args:
            preprocessor: Preprocesador de Scikit-Learn.

        Returns:
            Pipeline completo (preprocesador + modelo ganador) entrenado con
            todo ``X_train`` e ``y_train``.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Los datos no han sido divididos.")

        models = self._define_models()
        cv = self._get_cv_splitter(n_splits=5)
        scorer = self._get_cv_scoring_string()

        self.cv_scores = {}
        for name, estimator in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", estimator)]
            )
            scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
            )
            self.cv_scores[name] = scores.tolist()
            self.logger.info(
                "CV scores %s: mean=%.4f | std=%.4f | folds=%s",
                name,
                float(np.mean(scores)),
                float(np.std(scores)),
                scores.tolist(),
            )

        # Selección por Parsimonia (T-Pareada)
        simple_scores = np.array(self.cv_scores["linear"])
        complex_scores = np.array(self.cv_scores["gradient_boosting"])

        try:
            _, p_value = stats.ttest_rel(complex_scores, simple_scores)
            p_value = float(p_value) if p_value is not None else 1.0
        except Exception as exc:
            self.logger.warning("No se pudo calcular la prueba T pareada: %s", exc)
            p_value = 1.0

        winner = "gradient_boosting" if p_value <= 0.05 else "linear"
        self.selected_model_name = winner

        self.logger.info(
            "Modelo ganador por parsimonia: %s | p_value=%.4f",
            winner,
            p_value,
        )

        final_estimator = models[winner]
        self.model = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", final_estimator)]
        )
        self.model.fit(self.X_train, self.y_train)
        self.logger.info("Modelo ganador entrenado con todo el conjunto de entrenamiento")

        return self.model

    def _evaluate_and_shap(
        self,
        final_pipeline: Pipeline,
        task_type: str,
    ) -> Dict[str, Any]:
        """Evalúa el modelo en test y extrae el top 5 de variables SHAP.

        Args:
            final_pipeline: Pipeline entrenado (preprocesador + modelo).
            task_type: ``'classification'`` o ``'regression'``.

        Returns:
            Diccionario con métricas finales y top 5 SHAP.
        """
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("No hay datos de test disponibles")

        self.logger.info("Desbloqueando datos de test para evaluación final")
        y_pred = final_pipeline.predict(self.X_test)

        metrics: Dict[str, float] = {}
        if task_type == "classification":
            metrics["accuracy"] = float(accuracy_score(self.y_test, y_pred))
            metrics["precision"] = float(
                precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(self.y_test, y_pred, average="weighted")
            )

            try:
                if hasattr(final_pipeline, "predict_proba"):
                    y_prob = final_pipeline.predict_proba(self.X_test)
                    n_classes = len(np.unique(self.y_test))
                    if n_classes == 2:
                        roc = roc_auc_score(self.y_test, y_prob[:, 1])
                    else:
                        roc = roc_auc_score(
                            self.y_test,
                            y_prob,
                            multi_class="ovr",
                            average="weighted",
                        )
                    metrics["roc_auc"] = float(roc)
            except Exception as exc:
                self.logger.warning("No se pudo calcular ROC-AUC: %s", exc)

            self.logger.info(
                "Métricas finales - Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %s",
                metrics["accuracy"],
                metrics.get("precision", float("nan")),
                metrics.get("recall", float("nan")),
                metrics.get("f1", float("nan")),
                metrics.get("roc_auc", "N/A"),
            )
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(self.y_test, y_pred)))
            metrics["r2"] = float(r2_score(self.y_test, y_pred))
            self.logger.info(
                "Métricas finales - RMSE: %.4f | R2: %.4f",
                metrics["rmse"],
                metrics["r2"],
            )

        # SHAP
        top_shap: List[Dict[str, Any]] = []
        try:
            preprocessor = final_pipeline.named_steps["preprocessor"]
            model = final_pipeline.named_steps["model"]
            X_test_transformed = preprocessor.transform(self.X_test)

            if hasattr(model, "predict_proba"):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_test_transformed, check_additivity=False)
                except Exception:
                    X_sample = shap.utils.sample(X_test_transformed, 100) if hasattr(shap, "utils") else X_test_transformed[:100]
                    explainer = shap.Explainer(model.predict, X_sample)
                    shap_values = explainer(X_test_transformed[:100])
            else:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_test_transformed, check_additivity=False)
                except Exception:
                    X_sample = shap.utils.sample(X_test_transformed, 100) if hasattr(shap, "utils") else X_test_transformed[:100]
                    explainer = shap.Explainer(model.predict, X_sample)
                    shap_values = explainer(X_test_transformed[:100])

            # Feature names after preprocessing
            feature_names = self._get_feature_names_after_preprocessing(preprocessor)

            if shap_values.values.ndim == 3:
                # Multiclass: usar suma de valores absolutos por clase
                mean_abs_shap = np.abs(shap_values.values).sum(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            shap_importance = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values(by="mean_abs_shap", ascending=False)

            top_shap = shap_importance.head(5).to_dict(orient="records")
            self.logger.info("Top 5 SHAP: %s", top_shap)
        except Exception as exc:
            self.logger.warning("No se pudieron calcular valores SHAP: %s", exc)

        return {
            "metrics": metrics,
            "top_shap": top_shap,
        }

    def _generate_visualizations(
        self,
        X_train: pd.DataFrame,
        num_cols: List[str],
    ) -> Path:
        """Genera visualizaciones de análisis exploratorio.

        Crea el directorio ``data/03_output/figures/`` si no existe y genera
        un heatmap de correlación de las variables numéricas seleccionadas.
        Opcionalmente genera un scatter plot de la variable con menor p-value.

        Args:
            X_train: Conjunto de entrenamiento con las columnas finales.
            num_cols: Lista de columnas numéricas seleccionadas.

        Returns:
            Ruta del directorio donde se guardaron las figuras.
        """
        figures_dir = Path("data") / "03_output" / "figures"
        os.makedirs(figures_dir, exist_ok=True)

        numeric_df = X_train[num_cols].select_dtypes(include=[np.number]).dropna()

        if not numeric_df.empty and len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Heatmap de Correlación")
            plt.tight_layout()
            plt.savefig(figures_dir / "correlation.png", bbox_inches="tight")
            plt.close()
            self.logger.info("Heatmap de correlación guardado en: %s", figures_dir / "correlation.png")
        else:
            self.logger.warning("No hay suficientes columnas numéricas para generar heatmap de correlación")

        # Scatter plot opcional de la mejor variable numérica vs target
        if self.y_train is not None and num_cols:
            best_col = None
            lowest_p = float("inf")
            for col in num_cols:
                p = self.p_values.get(col, 1.0)
                if p < lowest_p:
                    lowest_p = p
                    best_col = col
            if best_col is not None:
                plt.figure(figsize=(8, 6))
                if self.task_type == "classification":
                    sns.boxplot(x=self.y_train, y=X_train[best_col])
                    plt.title(f"Boxplot de {best_col} vs Target")
                else:
                    sns.scatterplot(x=X_train[best_col], y=self.y_train, alpha=0.5)
                    plt.title(f"Scatter plot: {best_col} vs Target")
                plt.xlabel(best_col)
                plt.ylabel("Target")
                plt.tight_layout()
                plt.savefig(figures_dir / "scatter.png", bbox_inches="tight")
                plt.close()
                self.logger.info("Scatter/boxplot guardado en: %s", figures_dir / "scatter.png")

        return figures_dir

    def _generate_markdown_report(
        self,
        target_col: str,
        task_type: str,
        winner_name: Optional[str],
        metrics: Dict[str, float],
        shap_data: List[Dict[str, Any]],
    ) -> Path:
        """Genera y guarda un reporte ejecutivo en Markdown.

        Args:
            target_col: Columna objetivo.
            task_type: Tipo de tarea.
            winner_name: Nombre del modelo ganador.
            metrics: Métricas finales en test.
            shap_data: Top 5 variables SHAP.

        Returns:
            Ruta del archivo Markdown generado.
        """
        report_path = Path("data") / "03_output" / "reporte_ejecutivo.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats_report = getattr(self, "stats_report", {})
        hypothesis = stats_report.get("hypothesis", {})
        vif = stats_report.get("vif", {})

        lines: List[str] = [
            "# Reporte de Modelado Predictivo",
            "",
            f"**Fecha y hora de generación:** {now}",
            "",
            "## 1. Auditoría Estadística",
            "",
            "### 1.1 Pruebas de Hipótesis",
            "",
        ]

        if hypothesis:
            lines.extend([
                f"- **Total de columnas evaluadas:** {hypothesis.get('total', 'N/A')}",
                f"- **Umbral de p-value:** {hypothesis.get('threshold', 'N/A')}",
                f"- **Aprobadas:** {hypothesis.get('approved_count', 'N/A')} ({len(hypothesis.get('approved_columns', []))} columnas)",
                f"- **Rechazadas:** {hypothesis.get('rejected_count', 'N/A')} ({len(hypothesis.get('rejected_columns', []))} columnas)",
                "",
                "**Columnas aprobadas:**",
                "",
                ", ".join(f"`{c}`" for c in hypothesis.get("approved_columns", [])) or "N/A",
                "",
                "**Columnas rechazadas:**",
                "",
                ", ".join(f"`{c}`" for c in hypothesis.get("rejected_columns", [])) or "N/A",
                "",
            ])
        else:
            lines.append("No se ejecutaron pruebas de hipótesis.")
            lines.append("")

        lines.extend([
            "### 1.2 Filtro VIF (Colinealidad)",
            "",
        ])

        if vif:
            lines.extend([
                f"- **Umbral VIF:** {vif.get('threshold', 'N/A')}",
                f"- **Columnas numéricas originales:** {vif.get('original_numeric_count', 'N/A')}",
                f"- **Columnas numéricas sobrevivientes:** {vif.get('surviving_numeric_count', 'N/A')}",
                f"- **Columnas eliminadas por colinealidad:** {vif.get('removed_numeric_count', 'N/A')}",
                "",
                "**Columnas eliminadas:**",
                "",
                ", ".join(f"`{c}`" for c in vif.get("removed_columns", [])) or "Ninguna",
                "",
            ])
        else:
            lines.append("No se ejecutó filtro VIF.")
            lines.append("")

        lines.extend([
            "## 2. Análisis Visual",
            "",
            "### 2.1 Heatmap de Correlación",
            "",
            "![Heatmap de correlación](figures/correlation.png)",
            "",
        ])

        lines.extend([
            "## 3. Desempeño del Modelo",
            "",
            f"- **Target:** `{target_col}`",
            f"- **Tipo de tarea:** {task_type}",
            f"- **Modelo seleccionado:** {winner_name or 'N/A'}",
            "",
            "| Métrica | Valor |",
            "|---------|-------|",
        ])

        if task_type == "classification":
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                value = metrics.get(key)
                formatted = f"{value:.4f}" if value is not None else "N/A"
                lines.append(f"| {key.replace('_', ' ').title()} | {formatted} |")
        else:
            for key in ["rmse", "r2"]:
                value = metrics.get(key)
                formatted = f"{value:.4f}" if value is not None else "N/A"
                lines.append(f"| {key.upper()} | {formatted} |")

        shap_table = (
            "\n"
            "## 4. Explicabilidad (SHAP)\n"
            "\n"
            "Top 5 de variables que más impactan al modelo:\n"
            "\n"
            "| Posición | Nombre_Variable | Valor |\n"
            "|----------|-----------------|-------|\n"
        )

        for idx, row in enumerate(shap_data[:5], start=1):
            feature = row.get("feature", "N/A")
            importance = row.get("mean_abs_shap")
            formatted_imp = f"{float(importance):.6f}" if importance is not None else "N/A"
            shap_table += f"| {idx} | {feature} | {formatted_imp} |\n"

        lines.extend(shap_table.splitlines())

        lines.extend([
            "",
            "---",
            "*Reporte generado automáticamente por el Mega Software de ML.*",
            "",
        ])
        text_content = "\n".join(lines)
        for masked, original in self.column_mapping_inv.items():
            text_content = text_content.replace(masked, original)

        report_path.write_text(text_content, encoding="utf-8")
        self.logger.info("Reporte ejecutivo guardado en: %s", report_path.resolve())
        return report_path

    def _get_feature_names_after_preprocessing(
        self,
        preprocessor: ColumnTransformer,
    ) -> List[str]:
        """Obtiene los nombres de características tras el preprocesador.

        Args:
            preprocessor: ColumnTransformer ajustado.

        Returns:
            Lista de nombres de características.
        """
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            # Fallback: construir manualmente
            names = []
            for name, transformer, columns in preprocessor.transformers_:
                if name == "remainder":
                    continue
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend(transformer.get_feature_names_out(columns))
                else:
                    names.extend(columns)
            return names

    def _serialize_model(self, final_pipeline: Pipeline) -> Path:
        """Serializa el pipeline final en disco.

        Args:
            final_pipeline: Pipeline entrenado.

        Returns:
            Ruta del archivo serializado.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / "superv_model.pkl"
        joblib.dump(final_pipeline, model_path)
        self.logger.info("Modelo serializado en: %s", model_path.resolve())
        return model_path

    def _build_llm_payload(
        self,
        task_type: str,
        target_col: str,
        metrics: Dict[str, float],
        top_shap: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ensambla el payload para el LLM con resultados de la fase.

        Args:
            task_type: Tipo de tarea.
            target_col: Columna objetivo.
            metrics: Métricas finales en test.
            top_shap: Top 5 variables SHAP.

        Returns:
            Diccionario payload.
        """
        return {
            "pipeline_id": self.pipeline_id,
            "phase": "fase_1_superv",
            "task_type": task_type,
            "target_col": target_col,
            "selected_model": self.selected_model_name,
            "cv_scores": self.cv_scores,
            "test_metrics": metrics,
            "top_5_shap": top_shap,
        }

    def execute(
        self,
        target_col: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Orquestador principal de la Fase 1.

        Ejecuta el flujo completo: división, selección de variables, VIF,
        preprocesamiento, entrenamiento por parsimonia, evaluación SHAP,
        serialización del modelo y generación del payload para el LLM.

        Args:
            target_col: Nombre de la columna objetivo.
            task_type: ``'classification'`` o ``'regression'``.

        Returns:
            Diccionario con el reporte final de la fase.
        """
        self.logger.info("Iniciando Fase 1 - Ejecución completa")

        self.safe_execute(self._data_splitting, target_col, task_type)
        approved_cols = self.safe_execute(self._hypothesis_testing, target_col, task_type)

        numeric_cols = [
            c for c in approved_cols
            if self._is_numeric(self.X_train[c])
        ] if self.X_train is not None else []
        cat_cols = [
            c for c in approved_cols
            if not self._is_numeric(self.X_train[c])
        ] if self.X_train is not None else []

        vif_cols = self.safe_execute(self._vif_filter, numeric_cols)

        final_feature_cols = list(vif_cols) + cat_cols
        if self.X_train is not None and final_feature_cols:
            self.X_train = self.X_train[final_feature_cols]
            self.X_test = self.X_test[final_feature_cols]

        preprocessor = self._build_preprocessor(vif_cols, cat_cols)
        if not final_feature_cols:
            msg = "No quedaron columnas tras la selección. No se puede entrenar un modelo."
            self.logger.error(msg)
            raise RuntimeError(msg)

        final_pipeline = self.safe_execute(self._train_and_select_model, preprocessor)
        eval_result = self.safe_execute(self._evaluate_and_shap, final_pipeline, task_type)
        model_path = self.safe_execute(self._serialize_model, final_pipeline)

        payload = self._build_llm_payload(
            task_type=task_type,
            target_col=target_col,
            metrics=eval_result["metrics"],
            top_shap=eval_result["top_shap"],
        )

        # Guardar payload como checkpoint con DataFrame dummy
        dummy_df = pd.DataFrame({"report": [1]})
        self.checkpoint_manager.save_checkpoint(
            "fase_1_report",
            dummy_df,
            metadata=payload,
        )

        self.safe_execute(self._generate_visualizations, self.X_train, vif_cols)
        self.safe_execute(
            self._generate_markdown_report,
            target_col=target_col,
            task_type=task_type,
            winner_name=self.selected_model_name,
            metrics=eval_result["metrics"],
            shap_data=eval_result["top_shap"],
        )

        self.logger.info("Reporte Markdown y gráficos generados con éxito.")
        self.logger.info("Fase 1 - Ejecución completa finalizada")
        return {
            "task_type": task_type,
            "target_col": target_col,
            "approved_columns": approved_cols,
            "vif_numeric_columns": vif_cols,
            "selected_model": self.selected_model_name,
            "cv_scores": self.cv_scores,
            "test_metrics": eval_result["metrics"],
            "top_5_shap": eval_result["top_shap"],
            "model_path": str(model_path),
        }

    def run(
        self,
        target_col: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Ejecuta el flujo completo de la Fase 1 (sin evaluación final).

        Args:
            target_col: Nombre de la columna objetivo.
            task_type: ``'classification'`` o ``'regression'``.

        Returns:
            Diccionario con el estado final de la fase.
        """
        self.logger.info("Iniciando Fase 1 - Supervisado")

        self.safe_execute(self._data_splitting, target_col, task_type)
        approved_cols = self.safe_execute(self._hypothesis_testing, target_col, task_type)

        numeric_cols = [
            c for c in approved_cols
            if self._is_numeric(self.X_train[c])
        ] if self.X_train is not None else []
        cat_cols = [
            c for c in approved_cols
            if not self._is_numeric(self.X_train[c])
        ] if self.X_train is not None else []

        vif_cols = self.safe_execute(self._vif_filter, numeric_cols)

        final_feature_cols = list(vif_cols) + cat_cols
        if self.X_train is not None and final_feature_cols:
            self.X_train = self.X_train[final_feature_cols]
            self.X_test = self.X_test[final_feature_cols]

        preprocessor = self._build_preprocessor(vif_cols, cat_cols)
        if final_feature_cols:
            self.safe_execute(self._train_and_select_model, preprocessor)

        self.checkpoint_manager.save_checkpoint(
            "fase_1_selected",
            self.X_train,
            metadata={
                "pipeline_id": self.pipeline_id,
                "phase": "fase_1_superv",
                "task_type": task_type,
                "target_col": target_col,
                "approved_columns": approved_cols,
                "vif_numeric_columns": vif_cols,
                "selected_model": self.selected_model_name,
                "cv_scores": self.cv_scores,
                "train_rows": len(self.X_train) if self.X_train is not None else 0,
            },
        )

        self.logger.info("Fase 1 - Supervisado finalizada")
        return {
            "task_type": task_type,
            "target_col": target_col,
            "approved_columns": approved_cols,
            "vif_numeric_columns": vif_cols,
            "selected_model": self.selected_model_name,
            "cv_scores": self.cv_scores,
            "train_shape": self.X_train.shape if self.X_train is not None else (0, 0),
            "test_shape": self.X_test.shape if self.X_test is not None else (0, 0),
        }
