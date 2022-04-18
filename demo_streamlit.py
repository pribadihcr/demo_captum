import streamlit as st
from PIL import Image
import argparse
import os
import csv
import glob
import time
import logging
# from mxnet.ndarray.gen_op import LinearRegressionOutput
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import numpy as np

st.title('Industrial Visualization')

st.sidebar.header("User Input Image")

img = st.sidebar.file_uploader(label='Upload your bmp image', type=['bmp'])
if img:
    image = Image.open(img)
    st.image(image)

