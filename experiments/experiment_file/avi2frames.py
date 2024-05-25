import os
import sys

import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from datasets import load

success = load.avi2frames(os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C'))
print('Done.', success)