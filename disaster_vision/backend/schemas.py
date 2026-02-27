"""Pydantic schemas for DisasterVision API."""
from typing import List, Optional
from pydantic import BaseModel


class TileResult(BaseModel):
    tile_id: str                    # e.g. "tile_0"
    filename_pre: str
    filename_post: str
    damage_class: str               # no-damage | minor | major | destroyed
    confidence: float               # 0.0 – 1.0
    probability_distribution: dict  # {class: prob}
    priority_score: float           # severity_weight × confidence
    lat: Optional[float] = None     # centre lat of tile
    lon: Optional[float] = None     # centre lon of tile
    status: str = "ok"             # ok | error
    error_message: Optional[str] = None


class AnalysisResult(BaseModel):
    session_id: str
    total_tiles: int
    completed: int
    failed: int
    status: str                     # processing | done | error
    tiles: List[TileResult] = []

    # Aggregate stats (populated when status == "done")
    stats: Optional[dict] = None
    top_priority: Optional[List[TileResult]] = None
    situation_report: Optional[str] = None
