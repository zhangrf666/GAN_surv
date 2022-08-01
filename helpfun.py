import os
import time
import logging
import cv2
import random
import numpy as np
import pandas as pd

from PIL import Image
from visdom import Visdom
from sklearn.metrics import roc_curve
from lifelines.utils import concordance_index

def create_logger(log_dir):
    ''' Performs creating logger

    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    # logs settings
    log_file = os.path.join(log_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # initialize console handler
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger