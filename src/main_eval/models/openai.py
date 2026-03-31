"""
utils for OpenAI models
"""

import argparse 
import json 
from pathlib import Path 
from typing import Any, Dict, List 

import torch 
import numpy as np
from openai import OpenAI 
from PIL import Image
from tqdm import tqdm 

