"""
Pydantic Schemas for API Request/Response Models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ==================== Response Models ====================

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int


class Landmarks(BaseModel):
    """Facial landmarks"""
    left_eye: List[float]
    right_eye: List[float]
    nose: List[float]
    mouth_left: List[float]
    mouth_right: List[float]


class FaceDetection(BaseModel):
    """Single face detection result"""
    box: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score")
    landmarks: Optional[Landmarks] = None


class DetectFaceResponse(BaseModel):
    """Response for face detection endpoint"""
    success: bool
    faces_count: int
    faces: List[FaceDetection]
    image_size: Dict[str, int]


class RecognitionMatch(BaseModel):
    """Face recognition match result"""
    user_id: str
    name: Optional[str] = None
    similarity: float
    is_match: bool


class RecognizeFaceResponse(BaseModel):
    """Response for face recognition endpoint"""
    success: bool
    faces_detected: int
    matches: List[RecognitionMatch]
    message: Optional[str] = None


class TextureAnalysis(BaseModel):
    """Texture analysis results"""
    texture_variance: float
    texture_entropy: float
    texture_uniformity: float
    laplacian_variance: float
    texture_score: float
    is_real: bool


class ReflectionAnalysis(BaseModel):
    """Screen reflection analysis results"""
    center_energy: float
    outer_energy: float
    energy_ratio: float
    is_screen: bool


class ColorAnalysis(BaseModel):
    """Color distribution analysis results"""
    saturation_mean: float
    saturation_std: float
    cr_mean: float
    cr_std: float
    color_score: float
    is_natural: bool


class BlurAnalysis(BaseModel):
    """Blur detection results"""
    laplacian_variance: float
    gradient_magnitude: float
    sharpness_score: float
    is_sharp: bool


class AntiSpoofingChecks(BaseModel):
    """Anti-spoofing check details"""
    texture: Optional[TextureAnalysis] = None
    reflection: Optional[ReflectionAnalysis] = None
    color: Optional[ColorAnalysis] = None
    blur: Optional[BlurAnalysis] = None


class AntiSpoofingResponse(BaseModel):
    """Response for anti-spoofing endpoint"""
    success: bool
    is_live: bool
    confidence: float
    checks: AntiSpoofingChecks
    message: Optional[str] = None


# ==================== Request Models ====================

class AddFaceRequest(BaseModel):
    """Request for adding a face (used with form data)"""
    user_id: str = Field(..., description="Unique user identifier")
    name: Optional[str] = Field(None, description="Optional user name")


class UpdateFaceRequest(BaseModel):
    """Request for updating a face"""
    user_id: str = Field(..., description="User identifier to update")
    name: Optional[str] = Field(None, description="Optional new name")


# ==================== Database Response Models ====================

class FaceRecord(BaseModel):
    """Face record from database"""
    id: int
    user_id: str
    name: Optional[str]
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class GetFaceResponse(BaseModel):
    """Response for get face endpoint"""
    success: bool
    data: Optional[FaceRecord] = None
    message: Optional[str] = None


class AddFaceResponse(BaseModel):
    """Response for add face endpoint"""
    success: bool
    message: str
    user_id: Optional[str] = None
    id: Optional[int] = None


class UpdateFaceResponse(BaseModel):
    """Response for update face endpoint"""
    success: bool
    message: str


class DeleteFaceResponse(BaseModel):
    """Response for delete face endpoint"""
    success: bool
    message: str


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    database_users: int
