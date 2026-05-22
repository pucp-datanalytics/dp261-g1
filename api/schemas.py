"""Pydantic schemas aligned to handoff/contracts/ for Sprint 6."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BadBuyFeatures(BaseModel):
    """Raw vehicle features expected by the model API.

    The contract intentionally allows extra fields because the original dataset
    may contain auxiliary columns not used by the final preprocessing function.
    IsBadBuy should not be sent for production predictions.
    """

    PurchDate: Optional[str] = None
    Auction: Optional[str] = None
    VehYear: Optional[float] = None
    VehicleAge: float = Field(..., description="Vehicle age in years")
    Make: Optional[str] = None
    Model: Optional[str] = None
    Trim: Optional[str] = None
    SubModel: Optional[str] = None
    Color: Optional[str] = None
    Transmission: Optional[str] = None
    WheelTypeID: Optional[float] = None
    WheelType: Optional[str] = None
    VehOdo: float = Field(..., description="Vehicle odometer")
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
    VehBCost: float = Field(..., description="Vehicle acquisition cost")
    IsOnlineSale: Optional[Any] = None
    WarrantyCost: float = Field(..., description="Warranty cost")

    class Config:
        extra = "allow"


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    impact_direction: str


class BusinessValueAssumptions(BaseModel):
    benefit_tp: float = 2500
    cost_fp: float = -900
    cost_fn: float = 0
    benefit_tn: float = 0


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    threshold: float
    prediction: int = Field(..., ge=0, le=1)
    decision: str
    model_name: str
    model_version: str
    api_version: str
    business_value_assumptions: BusinessValueAssumptions
    shap_top_features: List[ShapFeature] = []
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
    model_sha: str
    threshold: float
