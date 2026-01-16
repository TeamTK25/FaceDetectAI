"""
API Routes for Face Recognition System

FastAPI endpoints for:
- Face detection
- Face recognition
- Anti-spoofing
- Face database management
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import numpy as np

from .schemas import (
    DetectFaceResponse, FaceDetection,
    RecognizeFaceResponse, RecognitionMatch,
    AntiSpoofingResponse, AntiSpoofingChecks, TextureAnalysis, ReflectionAnalysis,
    ColorAnalysis, BlurAnalysis,
    AddFaceResponse, GetFaceResponse, UpdateFaceResponse, DeleteFaceResponse,
    HealthResponse
)

from models.face_detector import get_face_detector
from models.face_recognizer import get_face_recognizer
from models.anti_spoofing import get_anti_spoofing
from models.database import get_face_database
from utils.image_utils import load_image_from_bytes

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    db = get_face_database()
    
    # Check model availability
    try:
        detector = get_face_detector()
        detector_available = detector.detector is not None
    except Exception:
        detector_available = False
    
    try:
        recognizer = get_face_recognizer()
        recognizer_available = recognizer.app is not None
    except Exception:
        recognizer_available = False
    
    try:
        anti_spoof = get_anti_spoofing()
        anti_spoof_available = True  # Always available now (no mediapipe dependency)
    except Exception:
        anti_spoof_available = False
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "face_detector": detector_available,
            "face_recognizer": recognizer_available,
            "anti_spoofing": anti_spoof_available
        },
        database_users=db.get_user_count()
    )


@router.post("/detect_face", response_model=DetectFaceResponse)
async def detect_face(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image
    
    Returns bounding boxes and facial landmarks for all detected faces.
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Detect faces
    detector = get_face_detector()
    
    try:
        detections = detector.detect_faces(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    # Format response
    faces = []
    for det in detections:
        face = FaceDetection(
            box=det['box'],
            confidence=det['confidence'],
            landmarks=det.get('landmarks')
        )
        faces.append(face)
    
    height, width = image.shape[:2]
    
    return DetectFaceResponse(
        success=True,
        faces_count=len(faces),
        faces=faces,
        image_size={"width": width, "height": height}
    )


@router.post("/recognize_face", response_model=RecognizeFaceResponse)
async def recognize_face(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Similarity threshold for matching")
):
    """
    Recognize faces in an uploaded image by comparing with database
    
    Returns best matches for each detected face.
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Get recognizer and database
    recognizer = get_face_recognizer()
    db = get_face_database()
    
    try:
        # Get embeddings from image
        face_data = recognizer.get_embedding_from_full_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
    
    if not face_data:
        return RecognizeFaceResponse(
            success=True,
            faces_detected=0,
            matches=[],
            message="No faces detected in image"
        )
    
    # Get all embeddings from database
    db_embeddings = db.get_all_embeddings()
    
    if not db_embeddings:
        return RecognizeFaceResponse(
            success=True,
            faces_detected=len(face_data),
            matches=[],
            message="No faces in database to compare with"
        )
    
    # Match each detected face
    matches = []
    for face in face_data:
        query_embedding = np.array(face['embedding'])
        result = recognizer.recognize(query_embedding, db_embeddings, threshold=threshold)
        
        if result:
            # Get name from database
            db_face = db.get_face(result['user_id'])
            name = db_face['name'] if db_face else None
            
            matches.append(RecognitionMatch(
                user_id=result['user_id'],
                name=name,
                similarity=result['similarity'],
                is_match=result['is_match']
            ))
    
    return RecognizeFaceResponse(
        success=True,
        faces_detected=len(face_data),
        matches=matches
    )


@router.post("/anti_spoofing", response_model=AntiSpoofingResponse)
async def check_anti_spoofing(file: UploadFile = File(...)):
    """
    Check if a face image is real or spoofed
    
    Performs texture analysis and screen reflection detection.
    For blink detection, use video frames endpoint (coming soon).
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Detect face first
    detector = get_face_detector()
    
    try:
        faces = detector.extract_faces(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")
    
    if not faces:
        return AntiSpoofingResponse(
            success=False,
            is_live=False,
            confidence=0.0,
            checks=AntiSpoofingChecks(),
            message="No face detected in image"
        )
    
    # Use the largest face
    face_image, _ = faces[0]
    
    # Run anti-spoofing checks
    anti_spoof = get_anti_spoofing()
    result = anti_spoof.check_liveness(face_image)
    
    # Build response
    checks = AntiSpoofingChecks()
    
    if 'texture' in result['checks']:
        t = result['checks']['texture']
        checks.texture = TextureAnalysis(**t)
    
    if 'reflection' in result['checks']:
        r = result['checks']['reflection']
        checks.reflection = ReflectionAnalysis(**r)
    
    if 'color' in result['checks']:
        c = result['checks']['color']
        checks.color = ColorAnalysis(**c)
    
    if 'blur' in result['checks']:
        b = result['checks']['blur']
        checks.blur = BlurAnalysis(**b)
    
    return AntiSpoofingResponse(
        success=True,
        is_live=result['is_live'],
        confidence=result['confidence'],
        checks=checks
    )


@router.post("/add_face", response_model=AddFaceResponse)
async def add_face(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    name: Optional[str] = Form(None)
):
    """
    Add a new face to the database
    
    Extracts face embedding from image and stores it with user_id.
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Get embedding
    recognizer = get_face_recognizer()
    
    try:
        face_data = recognizer.get_embedding_from_full_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
    
    if not face_data:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    # Use the first (or largest) face
    embedding = face_data[0]['embedding']
    
    # Store in database
    db = get_face_database()
    result = db.add_face(
        user_id=user_id,
        embedding=embedding,
        name=name
    )
    
    if not result['success']:
        raise HTTPException(status_code=409, detail=result['message'])
    
    return AddFaceResponse(
        success=True,
        message=result['message'],
        user_id=user_id,
        id=result.get('id')
    )


@router.get("/get_face/{user_id}", response_model=GetFaceResponse)
async def get_face(user_id: str):
    """
    Get face data for a user by their user_id
    """
    db = get_face_database()
    face = db.get_face(user_id)
    
    if face is None:
        return GetFaceResponse(
            success=False,
            data=None,
            message=f"User {user_id} not found"
        )
    
    return GetFaceResponse(
        success=True,
        data=face
    )


@router.delete("/delete_face/{user_id}", response_model=DeleteFaceResponse)
async def delete_face(user_id: str):
    """
    Delete a face from the database
    """
    db = get_face_database()
    result = db.delete_face(user_id)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['message'])
    
    return DeleteFaceResponse(
        success=True,
        message=result['message']
    )


@router.post("/update_face", response_model=UpdateFaceResponse)
async def update_face(
    file: UploadFile = File(None),
    user_id: str = Form(...),
    name: Optional[str] = Form(None)
):
    """
    Update face data for an existing user
    
    Can update embedding (by providing new image) and/or name.
    """
    db = get_face_database()
    
    # Check if user exists
    existing = db.get_face(user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    embedding = None
    
    # If new image provided, extract embedding
    if file is not None:
        contents = await file.read()
        
        try:
            image = load_image_from_bytes(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        recognizer = get_face_recognizer()
        
        try:
            face_data = recognizer.get_embedding_from_full_image(image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
        
        if not face_data:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        embedding = face_data[0]['embedding']
    
    # Update in database
    result = db.update_face(
        user_id=user_id,
        embedding=embedding,
        name=name
    )
    
    return UpdateFaceResponse(
        success=result['success'],
        message=result['message']
    )
