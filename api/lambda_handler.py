"""Adapter for running the FastAPI app on AWS Lambda with API Gateway."""
from mangum import Mangum

from api.main import app

handler = Mangum(app)
