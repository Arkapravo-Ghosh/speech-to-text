#!.venv/bin/python3

# Imports
from fastapi import FastAPI
import warnings
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from dotenv import load_dotenv

# Dotenv
try:
    load_dotenv(".env")
except:
    pass

# Route Imports
from app.routes.index_route import router as index_route
from app.routes.transcbribe_route import router as transcribe_route

# Config
warnings.filterwarnings("ignore")
app = FastAPI()
config = Config()
config.bind = ["localhost:6000"]


def main():
    # Routes
    app.include_router(index_route)
    app.include_router(prefix="/transcribe", router=transcribe_route)

    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
