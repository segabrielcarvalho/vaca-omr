from fastapi import FastAPI
from .routes.routes_v1 import router as api_router
import os

def create_app() -> FastAPI:
    app = FastAPI(
        title="VACA OMR API",
        version="1.0.0",
    )
    app.include_router(api_router, prefix="/api/v1")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "11001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
