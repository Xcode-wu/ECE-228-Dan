import os
import sys

import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath("../../")

sys.path.append(PROJECT_ROOT)
from datasets import load

result = load.avi2frames(os.path.join(PROJECT_ROOT, 'datasets', 'IIT-V2C'))
print('Done.', result)
