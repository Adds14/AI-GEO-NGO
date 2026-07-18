"""
FastAPI dependencies.
"""
from fastapi import Header, HTTPException

async def verify_api_key(api_key: str = Header(alias="X-API-Key", default="")):
    """
    Dependency to verify API Key.
    """
    if not api_key or api_key != "secret-ngo-token":
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key header. (Hint: Use 'secret-ngo-token')"
        )
    return api_key

def get_db():
    """Dependency to get a database session."""
    # db = SessionLocal()
    # try:
    #     yield db
    # finally:
    #     db.close()
    yield None
