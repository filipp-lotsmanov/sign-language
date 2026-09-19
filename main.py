"""
Entry point for the Sign Language Learning App.
"""

import logging
import os
import sys

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Bind to loopback by default. The WebSocket endpoint is unauthenticated, so
# binding 0.0.0.0 exposes model inference to everything on the network. Set
# HOST=0.0.0.0 explicitly when running in a container or behind a proxy.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


def main():
    logger.info("Starting Sign Language Learning App")

    try:
        import uvicorn

        from src.backend.api.routes import app

        logger.info("Starting server on http://%s:%s", HOST, PORT)

        # Single worker, deliberately. Sessions and the loaded models live in
        # process memory, so a second worker would hold a separate set of
        # sessions and requests would land on a worker that has never seen the
        # caller's session. Scaling out needs shared session storage first.
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            workers=1,
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
