from __future__ import annotations
import tkinter as tk
from tkinter import simpledialog
from PIL import Image
from tkinter import filedialog
from matplotlib import image
import numpy as np
from skimage.feature import hog
from skimage import exposure
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options

class HOGPersonDetector(ForensicsTool):
    def __init__(self): 
        self.tool_id = "hog_person_detector"
        self.title = "HOG Person Detector"
        self.category = "Set4"
        self.description = "Find people in an image using HOG features."
        self.requires_image = True
        