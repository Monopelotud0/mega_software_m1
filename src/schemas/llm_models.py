"""Esquemas Pydantic para interacciones estructuradas con LLMs.

Define modelos de respuesta que se usan como ``response_schema`` en las APIs
de Kimi (Moonshot AI) para forzar salidas tipadas y predecibles.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ColumnRecommendation(BaseModel):
    """Recomendación de preprocesamiento para una columna enmascarada."""

    column_name: str = Field(
        ...,
        description="Nombre genérico de la columna (ej. Var_01).",
    )
    action: str = Field(
        ...,
        description="Acción recomendada: 'impute_median', 'impute_mode', 'drop', 'scale', 'encode', 'keep'.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Breve justificación de la recomendación.",
    )


class MetadataResponse(BaseModel):
    """Respuesta estructurada del LLM con pasos de preprocesamiento."""

    overall_strategy: str = Field(
        ...,
        description="Estrategia general sugerida para el dataset.",
    )
    column_recommendations: List[ColumnRecommendation] = Field(
        ...,
        description="Lista de recomendaciones por columna.",
    )
    suggested_route: Optional[str] = Field(
        default=None,
        description="Ruta sugerida: 'fase_1', 'fase_2' o 'fase_3'.",
    )
