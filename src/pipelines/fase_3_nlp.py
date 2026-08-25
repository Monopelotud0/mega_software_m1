"""Fase 3: Procesamiento de Lenguaje Natural (NLP).

Este módulo implementa el pipeline [`NLPPipeline`](./src/pipelines/fase_3_nlp.py:18),
responsable de limpiar textos, lematizar con spaCy y vectorizar mediante TF-IDF
o embeddings densos de SentenceTransformer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import hdbscan
import numpy as np
import pandas as pd
import spacy
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipelines.base_pipeline import BasePipeline


class NLPPipeline(BasePipeline):
    """Pipeline de Procesamiento de Lenguaje Natural.

    Realiza limpieza de texto, lematización con spaCy y vectorización
    usando TF-IDF (sparse) o SentenceTransformer (dense).

    Args:
        pipeline_id: Identificador único de la ejecución del pipeline.
        params_path: Ruta opcional al archivo de configuración YAML.

    Attributes:
        nlp_model: Modelo spaCy cargado (puede ser ``None`` si falla).
    """

    def __init__(
        self,
        pipeline_id: str,
        params_path: Optional[str] = None,
    ) -> None:
        """Inicializa el pipeline NLP y carga el modelo spaCy disponible."""
        super().__init__(pipeline_id, params_path)
        self.nlp_model: Optional[spacy.Language] = None

        for model_name in ("es_core_news_sm", "en_core_web_sm"):
            try:
                self.nlp_model = spacy.load(model_name, disable=["parser", "ner"])
                self.logger.info("Modelo spaCy cargado: %s", model_name)
                break
            except OSError:
                self.logger.warning("Modelo spaCy no encontrado: %s", model_name)
            except Exception as exc:  # pragma: no cover - errores inesperados de carga
                self.logger.error("Error cargando spaCy %s: %s", model_name, exc)

        if self.nlp_model is None:
            self.logger.error(
                "No se pudo cargar ningún modelo spaCy. "
                "Ejecuta: python -m spacy download es_core_news_sm"
            )

    def _clean_text(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Limpia la columna de texto usando operaciones vectorizadas.

        Args:
            df: ``DataFrame`` con la columna de texto.
            text_col: Nombre de la columna a limpiar.

        Returns:
            ``DataFrame`` con una nueva columna ``<text_col>_clean``.
        """
        if text_col not in df.columns:
            raise ValueError(f"Columna '{text_col}' no encontrada en el DataFrame")

        cleaned = (
            df[text_col]
            .astype(str)
            .str.lower()
            .str.replace(r"http\S+|www\.\S+", " ", regex=True)
            .str.replace(r"\S+@\S+", " ", regex=True)
            .str.replace(r"[^a-záéíóúüñ0-9\s]", " ", regex=True)
            .str.replace(r"\b\d+\b", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        out = df.copy()
        out[f"{text_col}_clean"] = cleaned
        self.logger.info("Texto limpiado para columna: %s", text_col)
        return out

    def _spacy_process(self, texts: pd.Series) -> List[str]:
        """Lematiza textos usando spaCy en batches.

        Args:
            texts: Serie de textos limpios.

        Returns:
            Lista de textos lematizados.
        """
        if self.nlp_model is None:
            raise RuntimeError("Modelo spaCy no disponible para lematización")

        processed: List[str] = []
        for doc in self.nlp_model.pipe(texts.astype(str), batch_size=256, disable=["parser", "ner"]):
            lemmas = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and token.lemma_.strip()
            ]
            processed.append(" ".join(lemmas))

        self.logger.info("Lematización completada: %d textos", len(processed))
        return processed

    def _vectorize(self, texts: List[str], method: str = "dense") -> np.ndarray:
        """Vectoriza una lista de textos.

        Args:
            texts: Lista de strings a vectorizar.
            method: ``'dense'`` para SentenceTransformer o ``'sparse'`` para TF-IDF.

        Returns:
            Matriz de vectores como ``np.ndarray``.
        """
        if method not in {"dense", "sparse"}:
            raise ValueError("method debe ser 'dense' o 'sparse'")

        if method == "sparse":
            vectorizer = TfidfVectorizer(
                max_features=5000,
                max_df=0.90,
                min_df=5,
            )
            vectors = vectorizer.fit_transform(texts).toarray()
            self.logger.info(
                "Vectorización TF-IDF completada: shape=%s", vectors.shape
            )
            return vectors

        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vectors = model.encode(texts, batch_size=128, show_progress_bar=False)
            self.logger.info(
                "Vectorización densa completada: shape=%s", vectors.shape
            )
            return vectors
        except Exception as exc:
            self.logger.error(
                "Error en vectorización densa (posible OOM): %s. "
                "Recomendación: usa method='sparse'.",
                exc,
            )
            raise

    def _reduce_dimensions(self, vectors: np.ndarray) -> np.ndarray:
        """Reduce la dimensionalidad de los vectores con UMAP.

        Args:
            vectors: Matriz de vectores de alta dimensionalidad.

        Returns:
            Matriz reducida o los vectores originales si ya son de baja
            dimensionalidad.
        """
        if vectors.shape[1] < 10:
            self.logger.info(
                "Dimensionalidad ya es baja (%d), se omite UMAP", vectors.shape[1]
            )
            return vectors

        umap_components = self.config.get_nlp_param("umap_components", 5)
        reducer = umap.UMAP(
            n_components=umap_components,
            n_neighbors=15,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(vectors)
        self.logger.info(
            "UMAP completado: %s -> %s", vectors.shape, reduced.shape
        )
        return reduced

    def _cluster_topics(self, reduced_vectors: np.ndarray) -> np.ndarray:
        """Agrupa vectores reducidos en tópicos semánticos con HDBSCAN.

        Args:
            reduced_vectors: Matriz de vectores reducidos.

        Returns:
            Array con etiquetas de clúster. -1 indica ruido.
        """
        min_cluster_size = self.config.get_nlp_param("hdbscan_min_cluster_size", 15)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced_vectors)

        n_clusters = len(set(labels) - {-1})
        noise_pct = (labels == -1).sum() / len(labels) * 100
        self.logger.info(
            "HDBSCAN completado: tópicos=%d | ruido=%.2f%%", n_clusters, noise_pct
        )
        return labels

    def _extract_keywords(self, df_clustered: pd.DataFrame, text_col: str) -> Dict[int, List[str]]:
        """Extrae palabras clave por tópico usando c-TF-IDF simulado.

        Args:
            df_clustered: ``DataFrame`` con columna ``Topic_ID``.
            text_col: Nombre de la columna de texto limpio/lemmatizado.

        Returns:
            Diccionario ``topic_id -> lista de 10 palabras clave``.
        """
        valid_df = df_clustered[df_clustered["Topic_ID"] != -1]
        if valid_df.empty:
            self.logger.warning("No hay tópicos válidos para extraer keywords")
            return {}

        mega_docs = (
            valid_df.groupby("Topic_ID")[text_col]
            .apply(lambda x: " ".join(x.astype(str)))
            .sort_index()
        )

        vectorizer = TfidfVectorizer(max_features=10)
        tfidf = vectorizer.fit_transform(mega_docs)
        feature_names = vectorizer.get_feature_names_out()

        keywords: Dict[int, List[str]] = {}
        for idx, topic_id in enumerate(mega_docs.index):
            scores = tfidf[idx].toarray().flatten()
            top_indices = scores.argsort()[::-1][:10]
            keywords[int(topic_id)] = [feature_names[i] for i in top_indices]

        self.logger.info("Keywords extraídas para %d tópicos", len(keywords))
        return keywords

    def _merge_similar_topics(
        self,
        df_clustered: pd.DataFrame,
        vectors: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Fusiona tópicos cuyos centroides sean muy similares.

        Args:
            df_clustered: ``DataFrame`` con columna ``Topic_ID``.
            vectors: Vectores usados para calcular centroides.
            labels: Etiquetas de clúster actuales.

        Returns:
            Etiquetas actualizadas.
        """
        threshold = self.config.get_param("nlp.cosine_similarity_merge", 0.85)
        unique_labels = sorted([label for label in set(labels) if label != -1])
        if len(unique_labels) < 2:
            return labels

        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = vectors[mask].mean(axis=0)
            centroids.append(centroid)

        sim_matrix = cosine_similarity(np.vstack(centroids))
        merged_labels = labels.copy()

        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                if sim_matrix[i, j] > threshold:
                    source = unique_labels[i]
                    target = unique_labels[j]
                    minor, major = sorted([source, target])
                    merged_labels[merged_labels == minor] = major
                    self.logger.info(
                        "Fusionando tópico %d en %d (similitud=%.4f)",
                        minor,
                        major,
                        sim_matrix[i, j],
                    )

        return merged_labels

    def _build_payload(
        self,
        df_clustered: pd.DataFrame,
        keywords: Dict[int, List[str]],
        text_col: str,
    ) -> Dict[int, Dict[str, Any]]:
        """Construye el payload para que el LLM nombre cada tópico.

        Args:
            df_clustered: ``DataFrame`` con columna ``Topic_ID`` y texto original.
            keywords: Diccionario ``topic_id -> palabras clave``.
            text_col: Nombre de la columna de texto original.

        Returns:
            Payload por tópico con keywords y 3 frases de contexto.
        """
        payload: Dict[int, Dict[str, Any]] = {}
        rng = np.random.default_rng(seed=42)

        for topic_id, words in keywords.items():
            samples = (
                df_clustered[df_clustered["Topic_ID"] == topic_id][text_col]
                .dropna()
                .astype(str)
            )
            context = (
                samples.sample(n=min(3, len(samples)), random_state=rng.bit_generator)
                .tolist()
                if not samples.empty
                else []
            )
            payload[int(topic_id)] = {"keywords": words, "context": context}

        self.logger.info("Payload construido para %d tópicos", len(payload))
        return payload

    def execute(
        self,
        text_col: str,
        use_dense: bool = True,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Punto de entrada unificado de la Fase 3 NLP.

        Args:
            text_col: Columna de texto a procesar.
            use_dense: Si es ``True`` usa embeddings densos, si no TF-IDF.
            target_col: Si no es ``None``, redirige a clasificación (Fase 1).

        Returns:
            Diccionario con ``df``, ``vectors`` y metadatos/payload según ruta.
        """
        df, _ = self.checkpoint_manager.load_checkpoint("fase_0_clean")
        if df is None:
            raise RuntimeError("No se encontró checkpoint de fase_0_clean")

        method = "dense" if use_dense else "sparse"
        df_clean = self.safe_execute(self._clean_text, df, text_col)

        clean_col = f"{text_col}_clean"
        lemmas = self.safe_execute(self._spacy_process, df_clean[clean_col])
        df_clean["lemmatized"] = lemmas

        vectors = self.safe_execute(self._vectorize, lemmas, method)

        if target_col is not None:
            self.logger.info("Redirigiendo a Fase 1 (clasificación NLP)")
            metadata = {
                "text_col": text_col,
                "clean_col": clean_col,
                "target_col": target_col,
                "vector_method": method,
                "vector_shape": list(vectors.shape),
                "n_rows": len(df_clean),
            }
            self.df = df_clean
            self.metadata = metadata
            self.checkpoint_manager.save_checkpoint(
                "fase_3_nlp_vectors", df_clean, metadata
            )
            return {"df": df_clean, "vectors": vectors, "metadata": metadata}

        reduced = self.safe_execute(self._reduce_dimensions, vectors)
        labels = self.safe_execute(self._cluster_topics, reduced)
        labels = self.safe_execute(self._merge_similar_topics, df_clean, vectors, labels)
        df_clean["Topic_ID"] = labels

        keywords = self.safe_execute(self._extract_keywords, df_clean, "lemmatized")
        payload = self.safe_execute(self._build_payload, df_clean, keywords, text_col)

        metadata = {
            "text_col": text_col,
            "clean_col": clean_col,
            "vector_method": method,
            "vector_shape": list(vectors.shape),
            "reduced_shape": list(reduced.shape),
            "n_topics": int(len(set(labels) - {-1})),
            "noise_pct": float((labels == -1).sum() / len(labels) * 100),
            "n_rows": len(df_clean),
        }

        self.df = df_clean
        self.metadata = metadata

        self.checkpoint_manager.save_checkpoint("fase_3_payload", pd.DataFrame({"dummy": [1]}), metadata=payload)
        self.checkpoint_manager.save_checkpoint("fase_3_final", df_clean, metadata=metadata)
        self.logger.info("Fase 3 NLP (topic modeling) completada: %s", metadata)

        return {
            "df": df_clean,
            "vectors": reduced,
            "cluster_labels": labels,
            "keywords": keywords,
            "payload": payload,
            "metadata": metadata,
        }

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        text_col: str = "text",
        method: str = "dense",
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de NLP.

        Args:
            df: ``DataFrame`` opcional. Si es ``None`` intenta cargar el
                checkpoint de la fase anterior.
            text_col: Columna de texto a procesar.
            method: Método de vectorización (``'dense'`` o ``'sparse'``).

        Returns:
            Diccionario con ``df`` limpio, ``vectors`` reducidos,
            ``cluster_labels`` y ``metadata``.
        """
        if df is None:
            df, _ = self.checkpoint_manager.load_checkpoint("fase_2_unsuperv")
            if df is None:
                raise RuntimeError("No se encontró checkpoint de fase_2_unsuperv")

        df_clean = self.safe_execute(self._clean_text, df, text_col)

        clean_col = f"{text_col}_clean"
        lemmas = self.safe_execute(self._spacy_process, df_clean[clean_col])
        df_clean["lemmatized"] = lemmas

        vectors = self.safe_execute(self._vectorize, lemmas, method)
        reduced = self.safe_execute(self._reduce_dimensions, vectors)
        labels = self.safe_execute(self._cluster_topics, reduced)
        df_clean["topic"] = labels

        metadata = {
            "text_col": text_col,
            "clean_col": clean_col,
            "vector_method": method,
            "vector_shape": list(vectors.shape),
            "reduced_shape": list(reduced.shape),
            "n_topics": int(len(set(labels) - {-1})),
            "noise_pct": float((labels == -1).sum() / len(labels) * 100),
            "n_rows": len(df_clean),
        }

        self.df = df_clean
        self.metadata = metadata

        self.checkpoint_manager.save_checkpoint("fase_3_nlp", df_clean, metadata)
        self.logger.info("Fase 3 NLP completada: %s", metadata)

        return {
            "df": df_clean,
            "vectors": reduced,
            "cluster_labels": labels,
            "metadata": metadata,
        }
