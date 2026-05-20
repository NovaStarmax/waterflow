from fastapi import FastAPI
from api.routes import predict, health

app = FastAPI()

app.include_router(predict.router, prefix="/api")
app.include_router(health.router, prefix="/api")