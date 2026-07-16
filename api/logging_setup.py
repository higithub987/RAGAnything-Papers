"""Make the app's own (`api.*`) loggers visible in the server logs.

Uvicorn's default logging config attaches handlers only to the `uvicorn*` loggers,
not to the root logger, so records from `api.*` loggers have nowhere to go: Python's
"last resort" handler prints WARNING+ to stderr but silently drops INFO/DEBUG. That
swallowed useful startup INFO -- e.g. `lightrag_scope.apply_scope_patches`'s
"Container scope patches applied to: ..." confirmation, which is emitted at INFO on
the `api.lightrag_scope` logger and so never appeared even though the patch ran.

`configure_api_logging()` attaches a single stderr handler to the `api` logger at
INFO with propagation turned off, so `api.*` INFO surfaces without double-logging the
libraries (lightrag / raganything / uvicorn) that manage their own handlers.
"""

import logging
import sys

_API_LOGGER_NAME = "api"
# Marker set on the handler this module installs, so a repeat call is a no-op even if
# other handlers were added to the `api` logger in between.
_MARKER_ATTR = "_api_logging_handler"


def configure_api_logging(level: int = logging.INFO) -> logging.Logger:
    """Route the `api` logger's records to stderr at `level` (default INFO). Idempotent.

    Called once at server startup (see api.main.lifespan). Safe to call again -- if this
    module already installed its handler, nothing is added.
    """
    logger = logging.getLogger(_API_LOGGER_NAME)
    logger.setLevel(level)
    # propagate=False keeps these records off the root logger, so a future
    # logging.basicConfig() (or a library that configures root) can't double-print them.
    logger.propagate = False
    if not any(getattr(h, _MARKER_ATTR, False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:     %(name)s: %(message)s")
        )
        setattr(handler, _MARKER_ATTR, True)
        logger.addHandler(handler)
    return logger
