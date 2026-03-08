from fastapi import APIRouter
from app.models.schemas import HealthOut
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthOut)
async def health_check():
    return HealthOut(
        status="ok",
        version="0.1.0",
        environment=settings.APP_ENV,
    )
