"""
Entry point for the Sign Language Learning App.
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Sign Language Learning App")

    try:
        import uvicorn
        from src.backend.api.routes import app

        logger.info("Starting server on http://localhost:8000")

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )

    except ImportError as e:
        logger.error("Import error: %s", e)
        logger.error("Install dependencies with: uv pip install -e .")
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()