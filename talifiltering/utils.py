import itertools
import json
import logging
import pathlib
from typing import Dict, Union

import defusedxml.ElementTree as ET
import numpy as np
from rich.logging import RichHandler


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    return logging.getLogger("rich")
