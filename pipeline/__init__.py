import os
from pipeline.utils import setup_logging

if "DISABLE_LOG_CONFIG" not in os.environ:
    setup_logging()
