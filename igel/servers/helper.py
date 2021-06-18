import logging
import os

logger = logging.getLogger(__name__)


def remove_temp_data_file(f):
    """
    remove temporary file, where request payload has been stored in order to be used by igel to generate predictions
    """
    # remove temp file:
    if os.path.exists(f):
        logger.info(f"removing temporary file: {f}")
        os.remove(f)
