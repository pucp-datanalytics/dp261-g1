"""Pydantic schemas for the Bad Buy API."""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BadBuyFeatures(BaseModel):
    PurchDate: Optional[Any] = None
    Auction: Optional[str] = None
    VehYear: Optional[float] = None
    VehicleAge: Optional[float] = None
    Make: Optional[str] = None
    Model: Optional[str] = None
    Trim: Optional[str] = None
    SubModel: Optional[str] = None
    Color: Optional[str] = None
    Transmission: Optional[str] = None
    WheelTypeID: Optional[Any] = None
    WheelType: Optional[str] = None
    VehOdo: Optional[float] = None
    Nationality: Optional[str] = None
    Size: Optional[str] = None
    TopThreeAmericanName: Optional[str] = None
    MMRAcquisitionAuctionAveragePrice: Optional[float] = None
    MMRAcquisitionAuctionCleanPrice: Optional[float] = None
    MMRAcquisitionRetailAveragePrice: Optional[float] = None
    MMRAcquisitonRetailCleanPrice: Optional[float] = None
    MMRCurrentAuctionAveragePrice: Optional[float] = None
    MMRCurrentAuctionCleanPrice: Optional[float] = None
    MMRCurrentRetailAveragePrice: Optional[float] = None
    MMRCurrentRetailCleanPrice: Optional[float] = None
    PRIMEUNIT: Optional[str] = None
    AUCGUART: Optional[str] = None
    BYRNO: Optional[Any] = None
    VNZIP1: Optional[Any] = None
    VNST: Optional[str] = None
    VehBCost: Optional[float] = None
    IsOnlineSale: Optional[Any] = None
    WarrantyCost: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class ExplainFeature(BaseModel):
    feature: str
    display_name: Optional[str] = None
    value: Optional[Any] = None
    impact: float = 0.0
    impact_abs: Optional[float] = None
    impact_direction: str = "factor_relevante"
    detail: Optional[str] = None


class BusinessValueAssumptions(BaseModel):
    benefit_tp: float = 2500
    benefit_tn: float = 600
    cost_fp: float = -900
    cost_fn: float = -4500


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    threshold: float
    prediction: int = Field(..., ge=0, le=1)
    risk_segment: str
    decision: str
    model_name: str
    model_version: str
    api_version: str
    business_value_assumptions: BusinessValueAssumptions
    shap_top_features: List[ExplainFeature] = []
    explanation_method: str = "unavailable"


class BatchPredictionRequest(BaseModel):
    records: List[BadBuyFeatures]


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class VersionResponse(BaseModel):
    api_version: str
    model_version: str
    model_name: str
    model_sha: str
    threshold: float
