"""
Face Recognition System API

FastAPI application for face detection, recognition, and anti-spoofing.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router
from config import API_HOST, API_PORT

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="""
    Hệ thống nhận diện khuôn mặt với các tính năng:
    
    - **Face Detection**: Phát hiện khuôn mặt sử dụng MTCNN
    - **Face Recognition**: Nhận diện khuôn mặt với InsightFace (ArcFace)
    - **Anti-Spoofing**: Phát hiện giả mạo với texture analysis và liveness detection
    - **Face Management**: Quản lý cơ sở dữ liệu khuôn mặt
    
    ## Quy trình sử dụng:
    
    1. **Đăng ký khuôn mặt**: Sử dụng `POST /add_face` để thêm khuôn mặt vào database
    2. **Nhận diện**: Sử dụng `POST /recognize_face` để nhận diện khuôn mặt
    3. **Chống giả mạo**: Sử dụng `POST /anti_spoofing` để kiểm tra tính xác thực
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Face Recognition"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
