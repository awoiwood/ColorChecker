"""
ColorChecker - Advanced Color Correction Tool

A comprehensive color correction application that uses ColorChecker charts
to calibrate and correct color accuracy in images. Supports multiple correction
methods including polynomial regression, matrix transformation, and 3D LUT.

Features:
- Multiple correction algorithms (Polynomial, Matrix, LUT)
- Advanced optimization with robust regression and cross-validation
- Multi-scale and color space optimization
- Skin tone priority and greyscale optimization
- Real-time color difference calculation (CIEDE2000, CIEDE94, CIEDE76)
- Interactive GUI with zoom, pan, and before/after comparison
- Batch processing capabilities

Author: ColorChecker Development Team
License: MIT
"""

# Standard library imports
import os
import logging
import multiprocessing as mp
from functools import partial, lru_cache
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Third-party imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from skimage.color import rgb2lab, lab2rgb
from scipy.interpolate import griddata
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
import colour
import PyOpenColorIO as OCIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import io
import datetime

# Performance optimization imports
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - using standard NumPy operations")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    # Test GPU memory availability
    try:
        mempool = cp.get_default_memory_pool()
        CUPY_AVAILABLE = True
    except:
        CUPY_AVAILABLE = False
        print("CuPy GPU memory not available")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU operations")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this at the top of the file
DIAGNOSTIC_MODE = False  # Set to True to use synthetic patch data for debugging

# GPU-accelerated LUT functions
if CUPY_AVAILABLE:
    def gpu_lut_interpolation(image_rgb, lut):
        """GPU-accelerated LUT interpolation using CuPy."""
        try:
            # Transfer data to GPU
            image_gpu = cp.asarray(image_rgb, dtype=cp.float32)
            lut_gpu = cp.asarray(lut, dtype=cp.float32)
            
            lut_size = lut_gpu.shape[0]
            h, w, _ = image_gpu.shape
            
            # Create coordinate grids on GPU
            r_coords = image_gpu[:, :, 0] * (lut_size - 1)
            g_coords = image_gpu[:, :, 1] * (lut_size - 1)
            b_coords = image_gpu[:, :, 2] * (lut_size - 1)
            
            # Get integer and fractional parts
            r0 = cp.floor(r_coords).astype(cp.int32)
            g0 = cp.floor(g_coords).astype(cp.int32)
            b0 = cp.floor(b_coords).astype(cp.int32)
            
            r1 = cp.minimum(r0 + 1, lut_size - 1)
            g1 = cp.minimum(g0 + 1, lut_size - 1)
            b1 = cp.minimum(b0 + 1, lut_size - 1)
            
            # Fractional parts for interpolation
            r_frac = r_coords - r0
            g_frac = g_coords - g0
            b_frac = b_coords - b0
            
            # Get the 8 corners using advanced indexing
            c000 = lut_gpu[r0, g0, b0]
            c001 = lut_gpu[r0, g0, b1]
            c010 = lut_gpu[r0, g1, b0]
            c011 = lut_gpu[r0, g1, b1]
            c100 = lut_gpu[r1, g0, b0]
            c101 = lut_gpu[r1, g0, b1]
            c110 = lut_gpu[r1, g1, b0]
            c111 = lut_gpu[r1, g1, b1]
            
            # Interpolation weights
            w000 = (1 - r_frac) * (1 - g_frac) * (1 - b_frac)
            w001 = (1 - r_frac) * (1 - g_frac) * b_frac
            w010 = (1 - r_frac) * g_frac * (1 - b_frac)
            w011 = (1 - r_frac) * g_frac * b_frac
            w100 = r_frac * (1 - g_frac) * (1 - b_frac)
            w101 = r_frac * (1 - g_frac) * b_frac
            w110 = r_frac * g_frac * (1 - b_frac)
            w111 = r_frac * g_frac * b_frac
            
            # Interpolate
            corrected_rgb = (w000[..., cp.newaxis] * c000 + 
                            w001[..., cp.newaxis] * c001 + 
                            w010[..., cp.newaxis] * c010 + 
                            w011[..., cp.newaxis] * c011 +
                            w100[..., cp.newaxis] * c100 + 
                            w101[..., cp.newaxis] * c101 + 
                            w110[..., cp.newaxis] * c110 + 
                            w111[..., cp.newaxis] * c111)
            
            # Transfer back to CPU
            return cp.asnumpy(corrected_rgb)
            
        except Exception as e:
            logger.warning(f"GPU LUT interpolation failed: {e}, falling back to CPU")
            return None

    def gpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024):
        """GPU-accelerated LUT interpolation with tiled processing for large images."""
        try:
            h, w, _ = image_rgb.shape
            lut_gpu = cp.asarray(lut, dtype=cp.float32)
            
            # Calculate number of tiles
            tiles_h = (h + tile_size - 1) // tile_size
            tiles_w = (w + tile_size - 1) // tile_size
            
            corrected_rgb = np.zeros_like(image_rgb, dtype=np.float32)
            
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    # Calculate tile boundaries
                    y_start = th * tile_size
                    y_end = min((th + 1) * tile_size, h)
                    x_start = tw * tile_size
                    x_end = min((tw + 1) * tile_size, w)
                    
                    # Extract tile
                    tile = image_rgb[y_start:y_end, x_start:x_end].copy()
                    
                    # Process tile on GPU
                    tile_gpu = cp.asarray(tile, dtype=cp.float32)
                    lut_size = lut_gpu.shape[0]
                    
                    # Create coordinate grids for tile
                    r_coords = tile_gpu[:, :, 0] * (lut_size - 1)
                    g_coords = tile_gpu[:, :, 1] * (lut_size - 1)
                    b_coords = tile_gpu[:, :, 2] * (lut_size - 1)
                    
                    # Get integer and fractional parts
                    r0 = cp.floor(r_coords).astype(cp.int32)
                    g0 = cp.floor(g_coords).astype(cp.int32)
                    b0 = cp.floor(b_coords).astype(cp.int32)
                    
                    r1 = cp.minimum(r0 + 1, lut_size - 1)
                    g1 = cp.minimum(g0 + 1, lut_size - 1)
                    b1 = cp.minimum(b0 + 1, lut_size - 1)
                    
                    # Fractional parts
                    r_frac = r_coords - r0
                    g_frac = g_coords - g0
                    b_frac = b_coords - b0
                    
                    # Get corners
                    c000 = lut_gpu[r0, g0, b0]
                    c001 = lut_gpu[r0, g0, b1]
                    c010 = lut_gpu[r0, g1, b0]
                    c011 = lut_gpu[r0, g1, b1]
                    c100 = lut_gpu[r1, g0, b0]
                    c101 = lut_gpu[r1, g0, b1]
                    c110 = lut_gpu[r1, g1, b0]
                    c111 = lut_gpu[r1, g1, b1]
                    
                    # Interpolation weights
                    w000 = (1 - r_frac) * (1 - g_frac) * (1 - b_frac)
                    w001 = (1 - r_frac) * (1 - g_frac) * b_frac
                    w010 = (1 - r_frac) * g_frac * (1 - b_frac)
                    w011 = (1 - r_frac) * g_frac * b_frac
                    w100 = r_frac * (1 - g_frac) * (1 - b_frac)
                    w101 = r_frac * (1 - g_frac) * b_frac
                    w110 = r_frac * g_frac * (1 - b_frac)
                    w111 = r_frac * g_frac * b_frac
                    
                    # Interpolate
                    corrected_tile = (w000[..., cp.newaxis] * c000 + 
                                     w001[..., cp.newaxis] * c001 + 
                                     w010[..., cp.newaxis] * c010 + 
                                     w011[..., cp.newaxis] * c011 +
                                     w100[..., cp.newaxis] * c100 + 
                                     w101[..., cp.newaxis] * c101 + 
                                     w110[..., cp.newaxis] * c110 + 
                                     w111[..., cp.newaxis] * c111)
                    
                    # Transfer back to CPU and place in result
                    corrected_rgb[y_start:y_end, x_start:x_end] = cp.asnumpy(corrected_tile)
                    
                    # Clear GPU memory for this tile
                    del tile_gpu, corrected_tile
                    cp.get_default_memory_pool().free_all_blocks()
            
            return corrected_rgb
            
        except Exception as e:
            logger.warning(f"GPU tiled LUT interpolation failed: {e}, falling back to CPU")
            return None

else:
    def gpu_lut_interpolation(image_rgb, lut):
        """Fallback when GPU is not available."""
        return None
    
    def gpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024):
        """Fallback when GPU is not available."""
        return None

# CPU tiled processing function
def cpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024, num_workers=None):
    """CPU-based LUT interpolation with tiled processing and multiprocessing."""
    h, w, _ = image_rgb.shape
    
    # Calculate number of tiles
    tiles_h = (h + tile_size - 1) // tile_size
    tiles_w = (w + tile_size - 1) // tile_size
    
    corrected_rgb = np.zeros_like(image_rgb, dtype=np.float32)
    
    # Prepare tile data for multiprocessing
    tile_data = []
    for th in range(tiles_h):
        for tw in range(tiles_w):
            y_start = th * tile_size
            y_end = min((th + 1) * tile_size, h)
            x_start = tw * tile_size
            x_end = min((tw + 1) * tile_size, w)
            
            tile = image_rgb[y_start:y_end, x_start:x_end].copy()
            tile_data.append((tile, lut, (y_start, y_end, x_start, x_end)))
    
    # Process tiles in parallel
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(tile_data))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_lut_tile, tile_data))
    
    # Combine results
    for (y_start, y_end, x_start, x_end), corrected_tile in results:
        corrected_rgb[y_start:y_end, x_start:x_end] = corrected_tile
    
    return corrected_rgb

def process_lut_tile(tile_data):
    """Process a single tile for multiprocessing."""
    tile, lut, coords = tile_data
    y_start, y_end, x_start, x_end = coords
    
    lut_size = lut.shape[0]
    h, w, _ = tile.shape
    
    # Create coordinate grids
    r_coords = tile[:, :, 0] * (lut_size - 1)
    g_coords = tile[:, :, 1] * (lut_size - 1)
    b_coords = tile[:, :, 2] * (lut_size - 1)
    
    # Get integer and fractional parts
    r0 = np.floor(r_coords).astype(int)
    g0 = np.floor(g_coords).astype(int)
    b0 = np.floor(b_coords).astype(int)
    
    r1 = np.minimum(r0 + 1, lut_size - 1)
    g1 = np.minimum(g0 + 1, lut_size - 1)
    b1 = np.minimum(b0 + 1, lut_size - 1)
    
    # Fractional parts
    r_frac = r_coords - r0
    g_frac = g_coords - g0
    b_frac = b_coords - b0
    
    # Get corners
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]
    
    # Interpolation weights
    w000 = (1 - r_frac) * (1 - g_frac) * (1 - b_frac)
    w001 = (1 - r_frac) * (1 - g_frac) * b_frac
    w010 = (1 - r_frac) * g_frac * (1 - b_frac)
    w011 = (1 - r_frac) * g_frac * b_frac
    w100 = r_frac * (1 - g_frac) * (1 - b_frac)
    w101 = r_frac * (1 - g_frac) * b_frac
    w110 = r_frac * g_frac * (1 - b_frac)
    w111 = r_frac * g_frac * b_frac
    
    # Interpolate
    corrected_tile = (w000[..., np.newaxis] * c000 + 
                     w001[..., np.newaxis] * c001 + 
                     w010[..., np.newaxis] * c010 + 
                     w011[..., np.newaxis] * c011 +
                     w100[..., np.newaxis] * c100 + 
                     w101[..., np.newaxis] * c101 + 
                     w110[..., np.newaxis] * c110 + 
                     w111[..., np.newaxis] * c111)
    
    return (y_start, y_end, x_start, x_end), corrected_tile

def process_image_chunk_standalone(chunk_data):
    """
    Process a chunk of the image using 3D LUT interpolation.
    
    This function must be outside the class for multiprocessing to work.
    
    Args:
        chunk_data: Tuple containing (chunk, lut, grid_ranges, chunk_id)
            - chunk: Image chunk in LAB color space
            - lut: 3D lookup table for color correction
            - grid_ranges: LAB grid ranges for interpolation
            - chunk_id: Identifier for the chunk
    
    Returns:
        Tuple of (chunk_id, corrected_chunk) where corrected_chunk is the
        color-corrected version of the input chunk
    """
    try:
        chunk, lut, grid_ranges, chunk_id = chunk_data
        h, w, _ = chunk.shape
        corrected_chunk = np.zeros_like(chunk)
        
        L_range, a_range, b_range = grid_ranges
        
        for y in range(h):
            for x in range(w):
                lab = chunk[y, x]
                L, a, b = lab
                
                # Handle edge cases
                if (np.isnan(L) or np.isinf(L) or 
                    np.isnan(a) or np.isinf(a) or 
                    np.isnan(b) or np.isinf(b)):
                    corrected_chunk[y, x] = np.array([50.0, 0.0, 0.0])
                    continue
                
                # Find grid indices
                iL = max(0, min(len(L_range)-2, np.searchsorted(L_range, L, side='right') - 1))
                ia = max(0, min(len(a_range)-2, np.searchsorted(a_range, a, side='right') - 1))
                ib = max(0, min(len(b_range)-2, np.searchsorted(b_range, b, side='right') - 1))
                
                # Compute fractional position
                fL = (L - L_range[iL]) / (L_range[iL+1] - L_range[iL])
                fa = (a - a_range[ia]) / (a_range[ia+1] - a_range[ia])
                fb = (b - b_range[ib]) / (b_range[ib+1] - b_range[ib])
                
                # Handle division by zero
                if np.isnan(fL) or np.isinf(fL):
                    fL = 0.0
                if np.isnan(fa) or np.isinf(fa):
                    fa = 0.0
                if np.isnan(fb) or np.isinf(fb):
                    fb = 0.0
                
                # Get the 8 corners of the cube
                c = lut[iL:iL+2, ia:ia+2, ib:ib+2].reshape(8, 3)
                
                # Simple trilinear interpolation instead of tetrahedral
                out = (c[0] * (1-fL) * (1-fa) * (1-fb) +
                       c[1] * fL * (1-fa) * (1-fb) +
                       c[2] * (1-fL) * fa * (1-fb) +
                       c[3] * fL * fa * (1-fb) +
                       c[4] * (1-fL) * (1-fa) * fb +
                       c[5] * fL * (1-fa) * fb +
                       c[6] * (1-fL) * fa * fb +
                       c[7] * fL * fa * fb)
                
                # Ensure output is valid
                if np.any(np.isnan(out)) or np.any(np.isinf(out)):
                    corrected_chunk[y, x] = np.array([50.0, 0.0, 0.0])
                else:
                    corrected_chunk[y, x] = out
        
        return chunk_id, corrected_chunk
    except Exception as e:
        # Return a simple fallback if anything goes wrong
        chunk, lut, grid_ranges, chunk_id = chunk_data
        return chunk_id, np.zeros_like(chunk)

class ColorCorrectionApp:
    """
    Main application class for the ColorChecker color correction tool.
    
    Provides a comprehensive GUI for loading images, selecting ColorChecker patches,
    and applying various color correction algorithms including polynomial regression,
    matrix transformation, and 3D LUT methods.
    """
    
    def __init__(self, root):
        """
        Initialize the ColorCorrectionApp.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("ColorChecker - Advanced Color Correction Tool")
        self.root.geometry("1600x1000")
        
        # ColorChecker 24 reference values in LAB color space
        # These values represent the standard ColorChecker chart colors
        self.reference_colors_lab = {
            'Dark Skin': (37.54, 14.37, 14.92),
            'Light skin': (64.66, 19.27, 17.5),
            'Blue sky': (49.32, -3.82, -22.54),
            'Foliage': (43.46, -12.74, 22.72),
            'Blue flower': (54.94, 9.61, -24.79),
            'Bluish green': (70.48, -32.26, -0.37),
            'Orange': (62.73, 35.83, 56.5),
            'Purplish blue': (39.43, 10.75, -45.17),
            'Moderate red': (50.57, 48.64, 16.67),
            'Purple': (30.1, 22.54, -20.87),
            'Yellow green': (71.77, -24.13, 58.19),
            'Orange yellow': (71.51, 18.24, 67.37),
            'Blue': (28.37, 15.42, -49.8),
            'Green': (54.38, -39.72, 32.27),
            'Red': (42.43, 51.05, 28.62),
            'Yellow': (81.8, 2.67, 80.41),
            'Magenta': (50.63, 51.28, -14.12),
            'Cyan': (49.57, -29.71, -28.32),
            'White': (95.19, -1.03, 2.93),
            'Neutral 8': (81.29, -0.57, 0.44),
            'Neutral 6.5': (66.89, -0.75, -0.06),
            'Neutral 5': (50.76, -0.13, 0.14),
            'Neutral 3.5': (35.63, -0.46, -0.48),
            'Black': (20.64, 0.07, -0.46)
        }
        
        # Neutral patches for white balance calibration
        self.neutral_patches = ['White', 'Neutral 8', 'Neutral 6.5', 'Neutral 5', 'Neutral 3.5', 'Black']
        
        # Core processing options
        self.apply_white_balance = tk.BooleanVar(value=True)
        self.apply_luminance_normalization = tk.BooleanVar(value=False)
        self.correction_method_var = tk.StringVar(value="Matrix")  # Options: "LUT", "Matrix", "Polynomial"
        
        # Advanced polynomial regression options
        self.polynomial_degree_var = tk.StringVar(value="Auto")  # Options: "Auto", "1", "2", "3", "4"
        self.use_robust_regression = tk.BooleanVar(value=True)
        self.use_cross_validation = tk.BooleanVar(value=True)
        self.use_advanced_features = tk.BooleanVar(value=True)  # Re-enabled
        self.use_multi_scale = tk.BooleanVar(value=True)  # Re-enabled
        
        # Performance optimization options
        self.enable_caching = tk.BooleanVar(value=True)
        self.evaluate_quality = tk.BooleanVar(value=False)  # Disabled by default for speed
        self.use_parallel_processing = tk.BooleanVar(value=True)
        self.optimization_level = tk.StringVar(value="Balanced")  # Options: "Fast", "Balanced", "Quality"
        
        # LUT processing options
        self.lut_processing_method = tk.StringVar(value="Auto")  # Options: "Auto", "GPU", "CPU_Tiled", "CPU_Standard"
        
        # Store polynomial models for ICC profile generation
        self.current_polynomial_models = None
        self.current_polynomial_degree = None
        self.current_polynomial_transformer = None
        
        # Store matrix transformation for LUT generation
        self.current_matrix_transformation = None
        
        # Configure modern styling
        self.setup_styles()
        
        # Initialize UI
        self.setup_ui()
    
    def setup_styles(self):
        """Configure modern styling for the application."""
        style = ttk.Style()
        
        # Configure modern theme
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        
        # Configure buttons
        style.configure('Primary.TButton', 
                       font=('Arial', 10, 'bold'),
                       background='#3498db',
                       foreground='white',
                       padding=(10, 5))
        
        style.configure('Success.TButton',
                       font=('Arial', 10, 'bold'),
                       background='#27ae60',
                       foreground='white',
                       padding=(10, 5))
        
        style.configure('Warning.TButton',
                       font=('Arial', 10, 'bold'),
                       background='#f39c12',
                       foreground='white',
                       padding=(10, 5))
        
        # Configure frames
        style.configure('Card.TFrame', relief='solid', borderwidth=1)
        style.configure('Panel.TFrame', relief='groove', borderwidth=2)
        
        # Patch weighting strategies
        self.use_color_only_weighting = tk.BooleanVar(value=False)  # Exclude neutral patches
        self.use_greyscale_optimization = tk.BooleanVar(value=False)  # Optimize for neutral patches
        self.use_skin_tone_priority = tk.BooleanVar(value=False)  # Prioritize skin tone patches
        
        self.selected_colors = {}
        self.selected_rects = {}  # color_name: (x0, y0, x1, y1) in display coords
        self.original_image = None
        self.original_image_rgb = None
        self.corrected_image = None
        self.current_selection = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """
        Set up the main user interface with all controls and display areas.
        """
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main container
        main_container = ttk.Frame(self.root, padding="15")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Title
        title_label = ttk.Label(header_frame, text="ColorChecker Pro", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Main action buttons
        actions_frame = ttk.Frame(header_frame)
        actions_frame.pack(side=tk.RIGHT)
        
        ttk.Button(actions_frame, text="📁 Load Image", style='Primary.TButton', 
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions_frame, text="🎨 Apply Correction", style='Success.TButton', 
                  command=self.apply_correction).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions_frame, text="💾 Save Image", style='Primary.TButton', 
                  command=self.save_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions_frame, text="📊 Export Report", style='Warning.TButton', 
                  command=self.export_pdf_report).pack(side=tk.LEFT, padx=(0, 8))
        
        # Left sidebar - Color Selection
        left_sidebar = ttk.Frame(main_container, style='Panel.TFrame')
        left_sidebar.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        left_sidebar.columnconfigure(0, weight=1)
        left_sidebar.rowconfigure(2, weight=1)
        
        # Color selection header
        color_header = ttk.Label(left_sidebar, text="Color Selection", style='Header.TLabel')
        color_header.grid(row=0, column=0, pady=(10, 5), padx=10)
        
        # Instructions
        instructions = ttk.Label(left_sidebar, 
                               text="1. Click a color name below\n2. Click the corresponding patch in the image\n3. Repeat for all patches", 
                               style='Info.TLabel', justify=tk.LEFT)
        instructions.grid(row=1, column=0, pady=(0, 10), padx=10, sticky=tk.W)
        
        # Color list with scrollbar
        list_frame = ttk.Frame(left_sidebar)
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.color_listbox = tk.Listbox(list_frame, height=20, width=25, 
                                       font=('Arial', 10), selectmode=tk.SINGLE,
                                       activestyle='none', selectbackground='#3498db',
                                       selectforeground='white')
        color_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.color_listbox.yview)
        self.color_listbox.configure(yscrollcommand=color_scrollbar.set)
        
        self.color_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        color_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Populate color list
        for color_name in self.reference_colors_lab.keys():
            self.color_listbox.insert(tk.END, color_name)
        
        self.color_listbox.bind('<<ListboxSelect>>', self.on_color_select)
        
        # Selected color display
        selected_frame = ttk.LabelFrame(left_sidebar, text="Selected Color", padding="10")
        selected_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 10))
        
        self.selected_color_label = ttk.Label(selected_frame, text="No color selected", style='Info.TLabel')
        self.selected_color_label.pack()
        
        # Progress section
        progress_frame = ttk.LabelFrame(left_sidebar, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 10))

        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var, style='Info.TLabel')
        progress_label.pack(pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=200)
        self.progress_bar.pack(fill=tk.X)
        
        # Control buttons
        controls_frame = ttk.Frame(left_sidebar)
        controls_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Reset", command=self.reset_selections).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Color Checker Example", command=self.test_reference_colors).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Options", command=self.show_options_menu).pack(side=tk.LEFT)
        
        # Center - Image display
        image_container = ttk.Frame(main_container, style='Card.TFrame')
        image_container.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        image_container.columnconfigure(0, weight=1)
        image_container.rowconfigure(0, weight=1)
        
        # Image display area
        self.image_frame = ttk.Frame(image_container, relief=tk.SUNKEN, borderwidth=1)
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="📸 Load an image to begin", 
                                   font=('Arial', 14), foreground='#7f8c8d')
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image controls
        image_controls = ttk.Frame(image_container)
        image_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 10))
        
        # View controls
        view_frame = ttk.LabelFrame(image_controls, text="View Controls", padding="5")
        view_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Zoom controls
        zoom_frame = ttk.Frame(view_frame)
        zoom_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(zoom_frame, text="Zoom:", style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.zoom_image(1.25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.zoom_image(0.8)).pack(side=tk.LEFT, padx=2)
        
        # Pan controls
        pan_frame = ttk.Frame(view_frame)
        pan_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(pan_frame, text="Pan:", style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(pan_frame, text="↑", width=3, command=lambda: self.pan_image(0, 50)).pack(side=tk.LEFT, padx=1)
        ttk.Button(pan_frame, text="↓", width=3, command=lambda: self.pan_image(0, -50)).pack(side=tk.LEFT, padx=1)
        ttk.Button(pan_frame, text="←", width=3, command=lambda: self.pan_image(50, 0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(pan_frame, text="→", width=3, command=lambda: self.pan_image(-50, 0)).pack(side=tk.LEFT, padx=1)
        
        # View buttons
        view_buttons = ttk.Frame(view_frame)
        view_buttons.pack(side=tk.LEFT)
        
        ttk.Button(view_buttons, text="🔍 Fit View", command=self.fit_image_to_window).pack(side=tk.LEFT, padx=(0, 5))
        
        self.showing_corrected = True
        self.before_after_button = ttk.Button(view_buttons, text="👁️ Show Before", command=self.toggle_before_after)
        self.before_after_button.pack(side=tk.LEFT)
        
        # Export controls
        export_frame = ttk.LabelFrame(image_controls, text="Export/Import", padding="5")
        export_frame.pack(side=tk.RIGHT)
        
        # Export button
        self.export_lut_button = ttk.Button(export_frame, text="📤 Export 3D LUT", 
                                          style='Warning.TButton', command=self.show_export_menu)
        self.export_lut_button.pack(pady=(0, 5))
        
        # Import button
        self.import_lut_button = ttk.Button(export_frame, text="📥 Import 3D LUT", 
                                          style='Info.TButton', command=self.import_lut)
        self.import_lut_button.pack()
        

        
        # Right sidebar - Analysis
        right_sidebar = ttk.Frame(main_container, style='Panel.TFrame')
        right_sidebar.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_sidebar.columnconfigure(0, weight=1)
        right_sidebar.rowconfigure(1, weight=1)
        
        # Analysis header
        analysis_header = ttk.Label(right_sidebar, text="Color Analysis", style='Header.TLabel')
        analysis_header.grid(row=0, column=0, pady=(10, 5), padx=10)
        
        # Analysis content
        analysis_content = ttk.Frame(right_sidebar)
        analysis_content.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=(0, 10))
        analysis_content.columnconfigure(0, weight=1)
        analysis_content.rowconfigure(1, weight=1)
        
        # Color comparison button
        ttk.Button(analysis_content, text="🎨 Show Color Comparison", 
                  command=self.show_color_comparison).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Analysis text area
        self.offset_text = tk.Text(analysis_content, height=25, width=35, wrap=tk.WORD,
                                 font=('Consolas', 9), bg='#f8f9fa', fg='#2c3e50',
                                 relief=tk.SUNKEN, borderwidth=1)
        offset_scrollbar = ttk.Scrollbar(analysis_content, orient=tk.VERTICAL, command=self.offset_text.yview)
        self.offset_text.configure(yscrollcommand=offset_scrollbar.set)
        
        self.offset_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        offset_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.offset_text.config(state=tk.DISABLED)
        
        # Bind events
        self.image_label.bind('<Button-1>', self.on_image_press)
        self.image_label.bind('<B1-Motion>', self.on_image_drag)
        self.image_label.bind('<ButtonRelease-1>', self.on_image_release)
        self.image_label.bind('<Control-Button-1>', self.on_pan_start)
        self.image_label.bind('<Control-B1-Motion>', self.on_pan_move)
        self.image_label.bind('<Control-ButtonRelease-1>', self.on_pan_end)
        
        # Keyboard shortcuts
        self.root.bind('<KeyPress-plus>', lambda e: self.zoom_image(1.25))
        self.root.bind('<KeyPress-minus>', lambda e: self.zoom_image(0.8))
        self.root.bind('<KeyPress-r>', lambda e: self.fit_image_to_window())
        
        # Initialize view state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self._pan_start_xy = None

    def on_pan_start(self, event):
        self._pan_start_xy = (event.x, event.y)
        self._pan_start_offset = self.pan_offset.copy()

    def on_pan_move(self, event):
        if self._pan_start_xy is None:
            return
        dx = event.x - self._pan_start_xy[0]
        dy = event.y - self._pan_start_xy[1]
        self.pan_offset[0] = self._pan_start_offset[0] + dx
        self.pan_offset[1] = self._pan_start_offset[1] + dy
        self.update_display_image()

    def on_pan_end(self, event):
        self._pan_start_xy = None

    def pan_image(self, dx, dy):
        if self.original_image_rgb is None:
            return
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self.update_display_image()

    def zoom_image(self, factor):
        if self.original_image_rgb is None:
            return
        self.zoom_level = max(0.1, min(self.zoom_level * factor, 10.0))
        self.update_display_image()

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.update_display_image()

    def fit_image_to_window(self):
        """
        Automatically zoom to fit the image in the window with padding.
        """
        if self.original_image_rgb is None:
            return
        
        # Get the current frame size
        canvas_w = self.image_frame.winfo_width()
        canvas_h = self.image_frame.winfo_height()
        
        # Use default size if frame hasn't been rendered yet
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w = 600
            canvas_h = 400
        
        # Get image dimensions
        img_h, img_w = self.original_image_rgb.shape[:2]
        
        # Calculate zoom to fit image in window with some padding
        zoom_x = (canvas_w - 40) / img_w
        zoom_y = (canvas_h - 40) / img_h
        self.zoom_level = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%
        
        # Center the image
        self.pan_offset = [0, 0]
        
        self.update_display_image()

    def toggle_before_after(self):
        """
        Toggle between showing the corrected and original image.
        """
        self.showing_corrected = not self.showing_corrected
        if self.showing_corrected:
            self.before_after_button.config(text="Show Before")
        else:
            self.before_after_button.config(text="Show After")
        self.update_display_image()

    def update_display_image(self):
        """
        Re-render the image with current zoom and pan settings.
        """
        if self.original_image_rgb is None:
            return
        
        # Select image to display (corrected or original)
        if self.corrected_image is not None and self.showing_corrected:
            img = self.corrected_image
        else:
            img = self.original_image_rgb
            
        h, w = img.shape[:2]
        zoom = self.zoom_level
        
        # Get canvas dimensions
        canvas_w = self.image_frame.winfo_width()
        canvas_h = self.image_frame.winfo_height()
        
        # Use default size if frame hasn't been rendered yet
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w = 600
            canvas_h = 400
        
        # Apply zoom transformation
        new_w, new_h = int(w * zoom), int(h * zoom)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply pan transformation
        x_off, y_off = self.pan_offset
        
        # Create display canvas
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 220
        
        # Calculate visible region
        x0 = max(0, -x_off)
        y0 = max(0, -y_off)
        x1 = min(new_w, canvas_w - x_off)
        y1 = min(new_h, canvas_h - y_off)
        cx0 = max(0, x_off)
        cy0 = max(0, y_off)
        cx1 = cx0 + (x1 - x0)
        cy1 = cy0 + (y1 - y0)
        
        # Copy visible portion to canvas
        if x1 > x0 and y1 > y0:
            canvas[cy0:cy1, cx0:cx1] = img_resized[y0:y1, x0:x1]
        
        # Draw overlays for all selected rectangles
        from PIL import ImageDraw, ImageFont
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        for color_name, rect in self.selected_rects.items():
            # Convert original image coordinates to display coordinates
            orig_x0, orig_y0, orig_x1, orig_y1 = rect
            # Apply zoom and pan transformation
            zoom = self.zoom_level
            x_off, y_off = self.pan_offset
            
            # Transform coordinates
            disp_x0 = orig_x0 * zoom + x_off
            disp_y0 = orig_y0 * zoom + y_off
            disp_x1 = orig_x1 * zoom + x_off
            disp_y1 = orig_y1 * zoom + y_off
            
            # Only draw if rectangle is visible in the display area
            if (disp_x1 > 0 and disp_x0 < canvas_w and 
                disp_y1 > 0 and disp_y0 < canvas_h):
                # Clip to display area
                draw_x0 = max(0, disp_x0)
                draw_y0 = max(0, disp_y0)
                draw_x1 = min(canvas_w, disp_x1)
                draw_y1 = min(canvas_h, disp_y1)
                # Draw black outline rectangle
                draw.rectangle([draw_x0-1, draw_y0-1, draw_x1+1, draw_y1+1], outline='black', width=4)
                # Draw white rectangle inside
                draw.rectangle([draw_x0, draw_y0, draw_x1, draw_y1], outline='white', width=2)
                # Draw label (color name) as white with black outline
                label = color_name
                draw.text((draw_x0 + 2, draw_y0 + 2), label, font=font, fill='white', stroke_width=2, stroke_fill='black')
        
        self.display_image = pil_canvas
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not load image")
                self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.selected_colors = {}
                self.selected_rects = {} # Clear previous rectangles
                self.current_selection = None
                
                # Reset corrected image and before/after state for new image
                self.corrected_image = None
                self.showing_corrected = False  # Always show original image when loading new image
                
                # Clear LUT state for new image
                if hasattr(self, 'current_lut'):
                    self.current_lut = None
                    self.current_lut_filename = None
                
                self.progress_var.set(f"Loaded image: {os.path.basename(file_path)}")
                
                # Update before/after button text
                if hasattr(self, 'before_after_button'):
                    self.before_after_button.config(text="👁️ Show After")
                
                # Reset zoom and pan, then automatically fit image to window
                self.zoom_level = 1.0
                self.pan_offset = [0, 0]
                self.root.after(100, self.fit_image_to_window)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def resize_image_for_display(self, image, max_size=600):
        """
        Deprecated: Image resizing is now handled by update_display_image.
        
        Args:
            image: Input image
            max_size: Maximum display size (unused)
            
        Returns:
            PIL Image object
        """
        return Image.fromarray(image)
    
    def get_display_to_image_coords(self, x, y):
        """
        Map display (canvas) coordinates to original image coordinates.
        
        Args:
            x: Display x coordinate
            y: Display y coordinate
            
        Returns:
            Tuple of (img_x, img_y) in original image coordinates
        """
        zoom = self.zoom_level
        x_off, y_off = self.pan_offset
        img_x = int((x - x_off) / zoom)
        img_y = int((y - y_off) / zoom)
        img_height, img_width = self.original_image_rgb.shape[:2]
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))
        return img_x, img_y

    def on_color_select(self, event):
        if not self.color_listbox.curselection():
            return
        
        selection = self.color_listbox.curselection()[0]
        color_name = self.color_listbox.get(selection)
        self.current_selection = color_name
        
        # Highlight selected color
        self.color_listbox.selection_clear(0, tk.END)
        self.color_listbox.selection_set(selection)
        

        
        # Show saved value for this color if it exists
        if color_name in self.selected_colors:
            self.update_selected_color_display(color_name)
            self.progress_var.set(f"{color_name}: Already selected. Click to re-select or choose another color.")
        else:
            self.progress_var.set(f"Click on {color_name} in the image")
            # Clear the color display if no saved value
            self.selected_color_label.configure(text="No color selected", image="")
    
    def on_image_press(self, event):
        if self.original_image_rgb is None or not self.current_selection:
            return
        self.drag_start = (event.x, event.y)
        self.drag_rect = None
        self.drag_overlay = None

    def on_image_drag(self, event):
        if self.original_image_rgb is None or not self.current_selection or not hasattr(self, 'drag_start'):
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        # Ensure coordinates are in correct order for drawing
        draw_x0, draw_x1 = min(x0, x1), max(x0, x1)
        draw_y0, draw_y1 = min(y0, y1), max(y0, y1)
        # Draw rectangle overlay on the image
        display_img = self.display_image.copy()
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(display_img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        # Draw black outline rectangle
        draw.rectangle([draw_x0-1, draw_y0-1, draw_x1+1, draw_y1+1], outline='black', width=4)
        # Draw white rectangle inside
        draw.rectangle([draw_x0, draw_y0, draw_x1, draw_y1], outline='white', width=2)
        # Draw label as white with black outline
        label = self.current_selection
        draw.text((draw_x0 + 2, draw_y0 + 2), label, font=font, fill='white', stroke_width=2, stroke_fill='black')
        self.drag_overlay = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=self.drag_overlay)
        self.image_label.image = self.drag_overlay
        self.drag_rect = (x0, y0, x1, y1)

    def on_image_release(self, event):
        if self.original_image_rgb is None or not self.current_selection or not hasattr(self, 'drag_start'):
            return
        
        # Store the current selection before clearing it
        selected_color_name = self.current_selection
        
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        # Ensure coordinates are in correct order
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        # Map to original image coordinates using zoom/pan
        orig_x0, orig_y0 = self.get_display_to_image_coords(x0, y0)
        orig_x1, orig_y1 = self.get_display_to_image_coords(x1, y1)
        # Ensure coordinates are within bounds
        if orig_x1 <= orig_x0 or orig_y1 <= orig_y0:
            self.progress_var.set("Invalid selection area. Please drag a rectangle.")
            return
        
        # Get the color in the selected area with improved sampling
        region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
        
        # Use median instead of mean to avoid edge effects
        # Reshape to get all pixels and calculate median for each channel
        pixels = region.reshape(-1, 3)
        median_color_rgb = np.median(pixels, axis=0).astype(np.uint8)
        
        # Alternative: use the center pixel if the region is small
        if region.shape[0] <= 10 and region.shape[1] <= 10:
            center_y = region.shape[0] // 2
            center_x = region.shape[1] // 2
            center_color_rgb = region[center_y, center_x]
            # Use center pixel but validate it's not too different from median
            if np.all(np.abs(center_color_rgb - median_color_rgb) < 30):  # 30 RGB units tolerance
                avg_color_rgb = center_color_rgb
            else:
                avg_color_rgb = median_color_rgb
        else:
            avg_color_rgb = median_color_rgb
        
        # Convert RGB to LAB (standard scale)
        avg_color_lab = self.rgb_to_lab(avg_color_rgb)
        
        # Validate the selected color is reasonable
        ref_lab = self.reference_colors_lab[selected_color_name]
        lab_diff = np.abs(np.array(avg_color_lab) - np.array(ref_lab))
        
        # Check if the selected color is within reasonable bounds
        if np.any(lab_diff > [30, 50, 50]):  # L*, a*, b* tolerances
            warning_msg = f"Warning: {selected_color_name} selection may be inaccurate.\n"
            warning_msg += f"Selected LAB: {tuple(round(x,2) for x in avg_color_lab)}\n"
            warning_msg += f"Expected LAB: {tuple(round(x,2) for x in ref_lab)}\n"
            warning_msg += f"Difference: {tuple(round(x,2) for x in lab_diff)}\n"
            warning_msg += "Consider reselecting this patch.\n\n"
            
            self.offset_text.config(state=tk.NORMAL)
            self.offset_text.insert(tk.END, warning_msg)
            self.offset_text.see(tk.END)
            self.offset_text.config(state=tk.DISABLED)
        
        # Store the LAB color for the specific selected color
        self.selected_colors[selected_color_name] = avg_color_lab
        # Store the rectangle in original image coordinates for overlay
        self.selected_rects[selected_color_name] = (orig_x0, orig_y0, orig_x1, orig_y1)
        
        # Debug: Show measured and reference LAB values in the offset/console area
        ref_lab_std = self.reference_colors_lab[selected_color_name]
        measured_lab_std = avg_color_lab
        delta_std = tuple(measured_lab_std[i] - ref_lab_std[i] for i in range(3))
        
        # Calculate color accuracy score (lower is better)
        accuracy_score = np.sqrt(np.sum(np.array(delta_std)**2))
        
        msg = f"{selected_color_name}:\n"
        msg += f"  Measured LAB: {tuple(round(x,2) for x in measured_lab_std)}\n"
        msg += f"  Reference LAB: {tuple(round(x,2) for x in ref_lab_std)}\n"
        msg += f"  Delta: ({round(delta_std[0],2)}, {round(delta_std[1],2)}, {round(delta_std[2],2)})\n"
        msg += f"  Accuracy Score: {accuracy_score:.2f} (lower is better)\n\n"
        
        self.offset_text.config(state=tk.NORMAL)
        self.offset_text.insert(tk.END, msg)
        self.offset_text.see(tk.END)
        self.offset_text.config(state=tk.DISABLED)
        
        # Update display with the selected color info
        self.update_selected_color_display(selected_color_name)
        
        # Restore image display (will redraw overlays)
        self.update_display_image()
        
        # Move to the next color in the list
        color_names = list(self.reference_colors_lab.keys())
        try:
            idx = color_names.index(selected_color_name)
            next_idx = idx + 1
            if next_idx < len(color_names):
                self.current_selection = color_names[next_idx]
                self.color_listbox.selection_clear(0, tk.END)
                self.color_listbox.selection_set(next_idx)
                self.progress_var.set(f"Click on {self.current_selection} in the image")
            else:
                self.current_selection = None
                self.color_listbox.selection_clear(0, tk.END)
        except Exception:
            self.current_selection = None
            self.color_listbox.selection_clear(0, tk.END)
        
        # Update visual indicators for selected colors
        self.update_color_list_display()
        
        # Check if all colors are selected
        if len(self.selected_colors) == len(self.reference_colors_lab):
            self.progress_var.set("All colors selected! Ready to apply correction.")
        else:
            remaining = len(self.reference_colors_lab) - len(self.selected_colors)
            self.progress_var.set(f"{remaining} colors remaining. Select a color name, then drag a rectangle.")

    def update_color_list_display(self):
        """Update the visual display of the color list to show selected colors"""
        # For now, we'll just update the progress text to show selected colors
        selected_count = len(self.selected_colors)
        total_count = len(self.reference_colors_lab)
        self.progress_var.set(f"Selected: {selected_count}/{total_count} colors")

    def rgb_to_lab(self, rgb_color):
        """Convert RGB color to standard LAB color space using skimage."""
        # rgb_color: (3,) uint8, 0-255
        rgb_normalized = np.array(rgb_color, dtype=np.float32) / 255.0
        lab = rgb2lab(rgb_normalized.reshape(1, 1, 3)).reshape(3)
        return tuple(lab)

    def lab_to_rgb(self, lab_color):
        """Convert standard LAB color to RGB using skimage."""
        lab_arr = np.array(lab_color, dtype=np.float32).reshape(1, 1, 3)
        rgb = lab2rgb(lab_arr).reshape(3)
        rgb_scaled = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb_scaled

    def correct_white_balance(self, image_rgb):
        """Apply white balance correction using only the Neutral 5 patch (true neutral grey)."""
        if not self.apply_white_balance.get():
            return image_rgb
        
        # Use only Neutral 5 for white balance
        patch_name = 'Neutral 5'
        if patch_name in self.selected_colors:
            # Convert LAB to RGB for white balance calculation
            lab_color = self.selected_colors[patch_name]
            rgb_color = self.lab_to_rgb(lab_color).astype(np.float32) / 255.0
            neutral_measurement = rgb_color
            
            # Get reference RGB for this neutral patch
            ref_lab = self.reference_colors_lab[patch_name]
            ref_rgb = self.lab_to_rgb(ref_lab).astype(np.float32) / 255.0
            neutral_reference = ref_rgb
        else:
            # If not selected, skip white balance
            return image_rgb
        
        # Avoid division by zero
        neutral_reference = np.maximum(neutral_reference, 0.01)
        wb_factors = neutral_reference / neutral_measurement
        
        # Apply white balance
        image_float = image_rgb.astype(np.float32) / 255.0
        wb_corrected = image_float * wb_factors
        wb_corrected = np.clip(wb_corrected, 0, 1)
        
        return (wb_corrected * 255).astype(np.uint8)

    def normalize_luminance(self, measured_labs, reference_labs):
        """Normalize luminance (L*) of measured patches to match reference."""
        if not self.apply_luminance_normalization.get():
            return measured_labs, reference_labs
        
        # Calculate average L* difference
        l_diffs = []
        for i, (measured, reference) in enumerate(zip(measured_labs, reference_labs)):
            l_diff = measured[0] - reference[0]  # L* difference
            l_diffs.append(l_diff)
        
        if len(l_diffs) == 0:
            return measured_labs, reference_labs
        
        # Calculate average L* adjustment
        avg_l_adjustment = np.mean(l_diffs)
        
        # Apply L* normalization to measured values
        normalized_measured = []
        for measured in measured_labs:
            normalized_l = measured[0] - avg_l_adjustment
            normalized_l = np.clip(normalized_l, 0, 100)  # Keep L* in valid range
            normalized_measured.append((normalized_l, measured[1], measured[2]))
        
        return normalized_measured, reference_labs

    def apply_polynomial_correction(self, image_rgb, measured_labs, reference_labs):
        """
        Apply enhanced polynomial regression correction with advanced features:
        1. Adaptive polynomial degree selection (1-4)
        2. Cross-validation for overfitting prevention
        3. Robust regression with outlier handling
        4. Multi-scale polynomial approach
        5. Confidence-based patch weighting
        6. Color space-specific optimization
        7. Advanced feature engineering
        8. PERFORMANCE OPTIMIZED with caching and vectorization
        """
        try:
            start_time = time.time()

            # DIAGNOSTIC MODE: Use synthetic patch data for debugging
            if DIAGNOSTIC_MODE:
                print("[DIAGNOSTIC MODE] Using synthetic patch data for polynomial correction!")
                np.random.seed(42)
                n = 24
                measured_labs = np.zeros((n, 3), dtype=np.float32)
                reference_labs = np.zeros((n, 3), dtype=np.float32)
                measured_labs[:, 0] = np.linspace(20, 90, n) + np.random.normal(0, 0.5, n)
                measured_labs[:, 1] = np.linspace(-40, 40, n) + np.random.normal(0, 0.5, n)
                measured_labs[:, 2] = np.linspace(-40, 40, n) + np.random.normal(0, 0.5, n)
                reference_labs = measured_labs + np.array([2.0, 3.0, -1.5])
                print("Synthetic measured_labs (first 3):\n", measured_labs[:3])
                print("Synthetic reference_labs (first 3):\n", reference_labs[:3])
            else:
                # Debug real patch data
                measured_labs = np.array(measured_labs, dtype=np.float32)
                reference_labs = np.array(reference_labs, dtype=np.float32)
                print("=== REAL PATCH DATA DEBUG ===")
                print(f"Real data - Measured LAB range: {np.min(measured_labs):.1f}-{np.max(measured_labs):.1f}")
                print(f"Real data - Reference LAB range: {np.min(reference_labs):.1f}-{np.max(reference_labs):.1f}")
                print(f"Real data - Transformation range: {np.min(reference_labs - measured_labs):.1f}-{np.max(reference_labs - measured_labs):.1f}")
                print(f"Real data - Sample measured: {measured_labs[:3]}")
                print(f"Real data - Sample reference: {reference_labs[:3]}")
                print(f"Real data - Sample transformation: {reference_labs[:3] - measured_labs[:3]}")
                print(f"Real data - Transformation magnitude: {np.linalg.norm(reference_labs - measured_labs, axis=1)}")
                print(f"Real data - Average transformation magnitude: {np.mean(np.linalg.norm(reference_labs - measured_labs, axis=1)):.3f}")

            # Step 1: Collect Patch Data (cached)
            cache_key = self._get_correction_cache_key(measured_labs, reference_labs, "polynomial")
            # Clear cache to force recalculation with fixed polynomial correction
            if hasattr(self, '_correction_cache') and cache_key in self._correction_cache:
                del self._correction_cache[cache_key]
                logger.info("Cleared cached polynomial correction to force recalculation")
            
            # Step 2: Preprocess Input - Normalize LAB values (vectorized)
            measured_labs_array = np.array(measured_labs, dtype=np.float32)
            reference_labs_array = np.array(reference_labs, dtype=np.float32)
            
            # Vectorized normalization
            measured_labs_normalized = self._normalize_lab_vectorized(measured_labs_array)
            reference_labs_normalized = self._normalize_lab_vectorized(reference_labs_array)
            
            # Step 3: Advanced Polynomial Optimization (cached)
            logger.info("Starting advanced polynomial optimization...")
            transformation_labs = reference_labs_normalized - measured_labs_normalized
            
            # Debug: Check transformation values
            print(f"Transformation stats: min={np.min(transformation_labs):.6f}, max={np.max(transformation_labs):.6f}, mean={np.mean(transformation_labs):.6f}")
            print(f"Sample transformations: {transformation_labs[:3]}")
            
            # Check if transformation is too small and amplify if needed
            transformation_magnitude = np.mean(np.linalg.norm(transformation_labs, axis=1))
            print(f"Average transformation magnitude: {transformation_magnitude:.6f}")
            
            # If transformation is too small (< 0.001), amplify it to make it more visible
            if transformation_magnitude < 0.001:
                amplification_factor = 0.01 / transformation_magnitude  # Target 0.01 magnitude
                transformation_labs = transformation_labs * amplification_factor
                print(f"Transformation too small, amplified by factor {amplification_factor:.1f}")
                print(f"New transformation magnitude: {np.mean(np.linalg.norm(transformation_labs, axis=1)):.6f}")
            
            best_models, best_degree, best_weights, best_delta_e = self.optimize_polynomial_advanced(
                measured_labs_normalized, transformation_labs
            )
            
            # Store models for ICC profile generation
            self.current_polynomial_models = best_models
            self.current_polynomial_degree = best_degree
            
            # Store the polynomial transformer for consistent feature creation
            if best_degree > 1:
                from sklearn.preprocessing import PolynomialFeatures
                self.current_polynomial_transformer = PolynomialFeatures(degree=best_degree, include_bias=True)
                self.current_polynomial_transformer.fit(measured_labs_normalized)
            else:
                self.current_polynomial_transformer = None
            
            logger.info(f"Advanced optimization complete. Best degree: {best_degree}, Best average ΔE: {best_delta_e:.2f}")
            
            # Step 4: Apply Correction to New Pixels (optimized)
            corrected_rgb = self._apply_polynomial_correction_optimized(
                image_rgb, best_models, best_degree, measured_labs_normalized
            )
            
            # Cache the result
            if not hasattr(self, '_correction_cache'):
                self._correction_cache = {}
            self._correction_cache[cache_key] = {
                'models': best_models,
                'degree': best_degree,
                'transformer': self.current_polynomial_transformer,
                'method': 'polynomial'
            }
            
            # Evaluate correction quality (optional for performance)
            if self.evaluate_quality.get():
                logger.info("Evaluating polynomial correction quality...")
                image_lab = rgb2lab(image_rgb.astype(np.float32) / 255.0)
                quality_report = self.evaluate_polynomial_quality(
                    measured_labs_array, reference_labs_array, image_lab, best_models, best_degree,
                    delta_e_method='CIEDE2000'
                )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Polynomial correction completed in {elapsed_time:.2f} seconds")
            
            return corrected_rgb
            
        except Exception as e:
            logger.error(f"Advanced polynomial correction failed: {e}")
            return image_rgb
    
    def optimize_polynomial_advanced(self, measured_labs_normalized, transformation_labs):
        """
        Advanced polynomial optimization with multiple enhancement strategies.
        Returns the best models, degree, weights, and corresponding average ΔE.
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        n_patches = len(measured_labs_normalized)
        
        def calculate_confidence_weights(measured_labs, transformation_labs):
            """Calculate confidence weights based on patch characteristics."""
            weights = np.ones(n_patches)
            
            print("Patch weighting analysis:")
            for i in range(n_patches):
                L, a, b = measured_labs[i, 0], measured_labs[i, 1], measured_labs[i, 2]
                
                # Calculate color saturation
                saturation = np.sqrt(a**2 + b**2)
                
                # Patch name for this index
                patch_names = list(self.reference_colors_lab.keys())
                patch_name = patch_names[i] if i < len(patch_names) else ""
                
                # Skin tone priority mode: prioritize 'Light skin' and 'Dark Skin' but maintain balance
                if self.use_skin_tone_priority.get():
                    if patch_name in ["Light skin", "Dark Skin"]:
                        weights[i] = 3.0  # High weight for skin tones (but not extreme)
                        print(f"  Patch {i} ({patch_name}): SKIN TONE PRIORITY - HIGH WEIGHT")
                    else:
                        weights[i] = 0.8  # Reduced but still significant weight for others
                        print(f"  Patch {i} ({patch_name}): SKIN TONE PRIORITY - REDUCED WEIGHT")
                    continue
                
                # Check if this is a greyscale patch (whites, greys, black)
                is_greyscale = (saturation < 0.1) or (L > 0.85) or (L < 0.05)
                
                # Check if this is a skin tone patch
                is_skin_tone = False
                skin_tone_category = ""
                
                # Comprehensive skin tone detection for various ethnicities and lighting
                # Primary skin tone range (Caucasian to light Asian)
                if (0.4 <= L <= 0.8 and  # Good luminance range for skin
                    0.08 <= a <= 0.25 and  # Reddish component
                    0.1 <= b <= 0.35):     # Yellowish component
                    is_skin_tone = True
                    skin_tone_category = "PRIMARY"
                
                # Extended skin tone range (darker skin tones)
                elif (0.25 <= L <= 0.6 and  # Darker skin range
                      0.05 <= a <= 0.3 and   # Extended reddish range
                      0.05 <= b <= 0.4):     # Extended yellowish range
                    is_skin_tone = True
                    skin_tone_category = "EXTENDED"
                
                # Warm skin tone indicators (any warm, human-like colors)
                elif (0.3 <= L <= 0.85 and   # Broad luminance range
                      a > 0.05 and           # Slight reddish tint
                      b > 0.05 and           # Slight yellowish tint
                      abs(a - b) < 0.15):    # Balanced warm tone
                    is_skin_tone = True
                    skin_tone_category = "WARM"
                
                # Standard weighting logic (when skin tone priority is OFF)
                # Color-only weighting: exclude whites, greys, and black
                if self.use_color_only_weighting.get() and is_greyscale:
                    weights[i] = 0.0
                    print(f"  Patch {i}: EXCLUDED (greyscale - L={L:.3f}, sat={saturation:.3f})")
                    continue
                
                # Greyscale optimization: focus on whites, greys, and black
                if self.use_greyscale_optimization.get():
                    if is_greyscale:
                        # Boost greyscale patches
                        if L > 0.9:  # Pure white
                            weights[i] *= 2.0
                            print(f"  Patch {i}: PURE WHITE - boosting weight")
                        elif L > 0.7:  # Light grey
                            weights[i] *= 1.8
                            print(f"  Patch {i}: LIGHT GREY - boosting weight")
                        elif L < 0.1:  # Near black
                            weights[i] *= 1.5
                            print(f"  Patch {i}: NEAR BLACK - boosting weight")
                        else:  # Medium grey
                            weights[i] *= 1.6
                            print(f"  Patch {i}: MEDIUM GREY - boosting weight")
                    else:
                        # Reduce weight for colored patches
                        weights[i] *= 0.3
                        print(f"  Patch {i}: COLORED - reducing weight")
                else:
                    # Standard weighting logic
                    # Exclude black entirely (L < 0.05 in normalized space)
                    if L < 0.05:
                        weights[i] = 0.0
                        print(f"  Patch {i}: EXCLUDED (black - L={L:.3f})")
                        continue
                    
                    # Weight by color saturation (more saturated colors are more reliable)
                    weights[i] *= (1.0 + saturation * 0.8)  # Boost saturated colors more
                    
                    # Weight by luminance - heavily penalize whites and neutrals
                    if L > 0.85:  # Very bright/white patches
                        weights[i] *= 0.3  # Heavily reduce weight for whites
                    elif L > 0.75:  # Bright patches
                        weights[i] *= 0.5  # Reduce weight for bright patches
                    elif L < 0.15:  # Very dark patches (but not black)
                        weights[i] *= 0.6  # Reduce weight for very dark patches
                    elif 0.2 < L < 0.7:  # Good luminance range
                        weights[i] *= 1.3  # Boost mid-tones
                    
                    # Penalize neutral colors (low saturation)
                    if saturation < 0.1:  # Very neutral colors
                        weights[i] *= 0.4  # Heavily reduce weight for neutrals
                    elif saturation < 0.2:  # Neutral colors
                        weights[i] *= 0.7  # Reduce weight for neutrals
                    elif saturation > 0.5:  # Highly saturated colors
                        weights[i] *= 1.4  # Boost highly saturated colors
                    
                    # Special handling for pure whites (L > 0.9 and low saturation)
                    if L > 0.9 and saturation < 0.05:
                        weights[i] *= 0.1  # Minimal weight for pure whites
                    
                    # Special handling for pure grays (very low saturation)
                    if saturation < 0.02:
                        weights[i] *= 0.2  # Very low weight for pure grays
                    
                    # Special handling for skin tones - boost them significantly
                    if is_skin_tone:
                        if skin_tone_category == "PRIMARY":
                            weights[i] *= 2.5  # High boost for primary skin tones
                            print(f"  Patch {i}: PRIMARY SKIN TONE DETECTED - boosting weight")
                        elif skin_tone_category == "EXTENDED":
                            weights[i] *= 2.0  # Boost for darker skin tones
                            print(f"  Patch {i}: EXTENDED SKIN TONE DETECTED - boosting weight")
                        else:  # WARM
                            weights[i] *= 1.5  # Moderate boost for warm tones
                            print(f"  Patch {i}: WARM TONE DETECTED - boosting weight")
                
                # Log patch characteristics and final weight
                weight_category = "normal"
                if weights[i] == 0.0:
                    weight_category = "excluded"
                elif weights[i] < 0.5:
                    weight_category = "low"
                elif weights[i] > 1.5:
                    weight_category = "high"
                
                print(f"  Patch {i}: L={L:.3f}, sat={saturation:.3f}, weight={weights[i]:.3f} ({weight_category})")
            
            # Normalize weights
            total_weight = np.sum(weights)
            if total_weight > 0:
                weights = weights / total_weight * n_patches
            else:
                # Fallback if all weights are zero
                weights = np.ones(n_patches)
                weights = weights / np.sum(weights) * n_patches
            
            print(f"Final normalized weights: {weights}")
            return weights
        
        def create_advanced_features(X):
            """Create advanced polynomial features with interaction terms."""
            try:
                from sklearn.preprocessing import PolynomialFeatures
                
                # Check for NaN values and replace them
                if np.any(np.isnan(X)):
                    print("Warning: NaN values detected in input data, replacing with zeros")
                    X = np.nan_to_num(X, nan=0.0)
                
                features = []
                
                # Basic polynomial features
                for degree in [1, 2, 3]:
                    poly = PolynomialFeatures(degree=degree, include_bias=True)
                    features.append(poly.fit_transform(X))
                
                # Interaction features
                L, a, b = X[:, 0], X[:, 1], X[:, 2]
                interactions = np.column_stack([
                    L * a, L * b, a * b,  # 2-way interactions
                    L * a * b,  # 3-way interaction
                    L**2 * a, L**2 * b, a**2 * L, a**2 * b, b**2 * L, b**2 * a,  # Higher-order interactions
                    np.sin(L * np.pi), np.cos(a * np.pi), np.sin(b * np.pi),  # Trigonometric features
                    np.sqrt(np.abs(L)), np.sqrt(np.abs(a)), np.sqrt(np.abs(b))  # Root features
                ])
                features.append(interactions)
                
                result = np.hstack(features)
                
                # Check for NaN values in result and replace them
                if np.any(np.isnan(result)):
                    print("Warning: NaN values detected in feature matrix, replacing with zeros")
                    result = np.nan_to_num(result, nan=0.0)
                
                return result
                
            except Exception as e:
                print(f"Error creating advanced features: {e}")
                # Fallback to simple polynomial features
                poly = PolynomialFeatures(degree=2, include_bias=True)
                return poly.fit_transform(X)
        
        def evaluate_model_cv(X, y, degree, weights, cv_folds=3):
            """Evaluate model using cross-validation."""
            try:
                from sklearn.preprocessing import PolynomialFeatures
                
                # Check for NaN values and replace them
                if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                    print("Warning: NaN values detected in CV data, replacing with zeros")
                    X = np.nan_to_num(X, nan=0.0)
                    y = np.nan_to_num(y, nan=0.0)
                
                if degree == 1:
                    features = X
                else:
                    poly = PolynomialFeatures(degree=degree, include_bias=True)
                    features = poly.fit_transform(X)
                
                # Check for NaN values in features
                if np.any(np.isnan(features)):
                    print("Warning: NaN values detected in CV features, replacing with zeros")
                    features = np.nan_to_num(features, nan=0.0)
                
                # Try different regression methods
                models = [
                    LinearRegression(),
                    HuberRegressor(epsilon=1.35, max_iter=100),
                    RANSACRegressor(random_state=42, max_trials=100)
                ]
                
                best_score = float('inf')
                best_model = None
                
                for model in models:
                    try:
                        if hasattr(model, 'fit'):
                            # Weighted cross-validation score
                            cv_scores = cross_val_score(
                                model, features, y, 
                                cv=cv_folds, 
                                scoring='neg_mean_squared_error',
                                fit_params={'sample_weight': weights} if hasattr(model, 'sample_weight') else {}
                            )
                            score = -np.mean(cv_scores)
                            
                            if score < best_score:
                                best_score = score
                                best_model = model
                    except:
                        continue
                
                return best_model, best_score
                
            except Exception as e:
                print(f"CV evaluation failed: {e}")
                return None, float('inf')
        
        def objective_function_advanced(params):
            """Advanced objective function with multiple optimization targets."""
            try:
                degree = int(params[0])
                weight_factor = params[1]
                
                # Calculate confidence-based weights
                base_weights = calculate_confidence_weights(measured_labs_normalized, transformation_labs)
                weights = base_weights * weight_factor
                weights = weights / np.sum(weights) * n_patches
                
                # Evaluate with cross-validation
                models = []
                total_score = 0
                
                for i in range(3):  # L*, a*, b* channels
                    model, score = evaluate_model_cv(
                        measured_labs_normalized, 
                        transformation_labs[:, i], 
                        degree, weights
                    )
                    
                    if model is None:
                        return 1000.0
                    
                    models.append(model)
                    total_score += score
                
                # Calculate final ΔE
                avg_score = total_score / 3
                
                # Penalize high degrees to prevent overfitting
                complexity_penalty = degree * 0.1
                
                return avg_score + complexity_penalty
                
            except Exception as e:
                print(f"Advanced objective function failed: {e}")
                return 1000.0
        
        # Multi-strategy optimization
        best_result = None
        best_delta_e = float('inf')
        best_models = None
        best_degree = 2
        best_weights = np.ones(n_patches)
        
        # Strategy 1: Adaptive degree selection with cross-validation
        print("Strategy 1: Adaptive degree selection...")
        
        # Determine which degrees to try based on user settings
        if self.polynomial_degree_var.get() == "Auto":
            degrees_to_try = [1, 2, 3, 4]
        else:
            degrees_to_try = [int(self.polynomial_degree_var.get())]
        
        for degree in degrees_to_try:
            base_weights = calculate_confidence_weights(measured_labs_normalized, transformation_labs)
            
            models = []
            total_score = 0
            
            for i in range(3):  # L*, a*, b* channels
                if self.use_cross_validation.get():
                    model, score = evaluate_model_cv(
                        measured_labs_normalized, 
                        transformation_labs[:, i], 
                        degree, base_weights
                    )
                else:
                    # Simple fitting without cross-validation
                    from sklearn.preprocessing import PolynomialFeatures
                    
                    if degree == 1:
                        features = measured_labs_normalized
                    else:
                        poly = PolynomialFeatures(degree=degree, include_bias=True)
                        features = poly.fit_transform(measured_labs_normalized)
                    
                    if self.use_robust_regression.get():
                        model = HuberRegressor(epsilon=1.35, max_iter=100)
                    else:
                        model = LinearRegression()
                    
                    model.fit(features, transformation_labs[:, i], sample_weight=base_weights)
                    score = mean_squared_error(transformation_labs[:, i], model.predict(features))
                
                if model is None:
                    continue
                
                models.append(model)
                total_score += score
            
            if len(models) == 3:
                avg_score = total_score / 3
                if avg_score < best_delta_e:
                    best_delta_e = avg_score
                    best_models = models
                    best_degree = degree
                    best_weights = base_weights
                    best_result = (models, degree, base_weights, avg_score)
        
        print(f"Strategy 1 best: degree={best_degree}, ΔE={best_delta_e:.2f}")
        
        # Strategy 2: Advanced feature engineering
        if self.use_advanced_features.get():
            print("Strategy 2: Advanced feature engineering...")
            try:
                advanced_features = create_advanced_features(measured_labs_normalized)
                
                # Use robust regression with advanced features
                models_advanced = []
                total_score_advanced = 0
                
                for i in range(3):  # L*, a*, b* channels
                    if self.use_robust_regression.get():
                        model = HuberRegressor(epsilon=1.35, max_iter=100)
                    else:
                        model = LinearRegression()
                    
                    model.fit(advanced_features, transformation_labs[:, i])
                    
                    # Evaluate
                    y_pred = model.predict(advanced_features)
                    score = mean_squared_error(transformation_labs[:, i], y_pred)
                    total_score_advanced += score
                    models_advanced.append(model)
                
                avg_score_advanced = total_score_advanced / 3
                if avg_score_advanced < best_delta_e:
                    best_delta_e = avg_score_advanced
                    best_models = models_advanced
                    best_degree = -1  # Special flag for advanced features
                    best_weights = calculate_confidence_weights(measured_labs_normalized, transformation_labs)
                    best_result = (models_advanced, -1, best_weights, avg_score_advanced)
                    print(f"Strategy 2 improved: ΔE={avg_score_advanced:.2f}")
            
            except Exception as e:
                print(f"Strategy 2 failed: {e}")
        else:
            print("Strategy 2: Advanced features disabled by user")
        
        # Strategy 3: Multi-scale polynomial approach
        if self.use_multi_scale.get():
            print("Strategy 3: Multi-scale polynomial approach...")
            try:
                # Combine different degrees for different color regions
                models_multi = []
                total_score_multi = 0
                
                # Use degree 1 for L* (luminance is more linear)
                # Use degree 2 for a* and b* (chroma has more curvature)
                degrees = [1, 2, 2]
                
                for i, degree in enumerate(degrees):
                    if self.use_cross_validation.get():
                        model, score = evaluate_model_cv(
                            measured_labs_normalized, 
                            transformation_labs[:, i], 
                            degree, best_weights
                        )
                    else:
                        # Simple fitting without cross-validation
                        if degree == 1:
                            features = measured_labs_normalized
                        else:
                            poly = PolynomialFeatures(degree=degree, include_bias=True)
                            features = poly.fit_transform(measured_labs_normalized)
                        
                        if self.use_robust_regression.get():
                            model = HuberRegressor(epsilon=1.35, max_iter=100)
                        else:
                            model = LinearRegression()
                        
                        model.fit(features, transformation_labs[:, i], sample_weight=best_weights)
                        score = mean_squared_error(transformation_labs[:, i], model.predict(features))
                    
                    if model is None:
                        continue
                    
                    models_multi.append(model)
                    total_score_multi += score
                
                if len(models_multi) == 3:
                    avg_score_multi = total_score_multi / 3
                    if avg_score_multi < best_delta_e:
                        best_delta_e = avg_score_multi
                        best_models = models_multi
                        best_degree = -2  # Special flag for multi-scale
                        best_weights = calculate_confidence_weights(measured_labs_normalized, transformation_labs)
                        best_result = (models_multi, -2, best_weights, avg_score_multi)
                        print(f"Strategy 3 improved: ΔE={avg_score_multi:.2f}")
            
            except Exception as e:
                print(f"Strategy 3 failed: {e}")
        else:
            print("Strategy 3: Multi-scale approach disabled by user")
        
        # Strategy 4: Global optimization (only if auto degree is selected)
        if self.polynomial_degree_var.get() == "Auto":
            print("Strategy 4: Global optimization...")
            try:
                # Optimize degree and weight factor simultaneously
                bounds = [(1, 4), (0.5, 2.0)]  # degree, weight_factor
                
                result = differential_evolution(
                    objective_function_advanced,
                    bounds,
                    maxiter=30,
                    popsize=8,
                    disp=False
                )
                
                if result.success:
                    opt_degree = int(result.x[0])
                    opt_weight_factor = result.x[1]
                    
                    # Recalculate with optimal parameters
                    base_weights = calculate_confidence_weights(measured_labs_normalized, transformation_labs)
                    opt_weights = base_weights * opt_weight_factor
                    opt_weights = opt_weights / np.sum(opt_weights) * n_patches
                    
                    models_opt = []
                    total_score_opt = 0
                    
                    for i in range(3):
                        if self.use_cross_validation.get():
                            model, score = evaluate_model_cv(
                                measured_labs_normalized, 
                                transformation_labs[:, i], 
                                opt_degree, opt_weights
                            )
                        else:
                            # Simple fitting without cross-validation
                            if opt_degree == 1:
                                features = measured_labs_normalized
                            else:
                                poly = PolynomialFeatures(degree=opt_degree, include_bias=True)
                                features = poly.fit_transform(measured_labs_normalized)
                            
                            if self.use_robust_regression.get():
                                model = HuberRegressor(epsilon=1.35, max_iter=100)
                            else:
                                model = LinearRegression()
                            
                            model.fit(features, transformation_labs[:, i], sample_weight=opt_weights)
                            score = mean_squared_error(transformation_labs[:, i], model.predict(features))
                        
                        if model is None:
                            continue
                        
                        models_opt.append(model)
                        total_score_opt += score
                    
                    if len(models_opt) == 3:
                        avg_score_opt = total_score_opt / 3
                        if avg_score_opt < best_delta_e:
                            best_delta_e = avg_score_opt
                            best_models = models_opt
                            best_degree = opt_degree
                            best_weights = opt_weights
                            best_result = (models_opt, opt_degree, opt_weights, avg_score_opt)
                            print(f"Strategy 4 improved: degree={opt_degree}, ΔE={avg_score_opt:.2f}")
            
            except Exception as e:
                print(f"Strategy 4 failed: {e}")
        else:
            print("Strategy 4: Global optimization skipped (fixed degree selected)")
        
        # Final model fitting with best parameters
        if best_models is None:
            # Fallback to simple polynomial
            poly = PolynomialFeatures(degree=2, include_bias=True)
            features = poly.fit_transform(measured_labs_normalized)
            
            best_models = []
            for i in range(3):
                model = LinearRegression()
                model.fit(features, transformation_labs[:, i], sample_weight=best_weights)
                best_models.append(model)
            best_degree = 2
        
        print(f"Final optimization result: degree={best_degree}, ΔE = {best_delta_e:.2f}")
        
        # After all strategies, ensure best_models are fitted on the full data
        if best_models is not None:
            # Determine features for fitting
            if best_degree == 1:
                features = measured_labs_normalized
            elif best_degree > 1:
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=best_degree, include_bias=True)
                features = poly.fit_transform(measured_labs_normalized)
            else:
                # For advanced features or multi-scale, use advanced features
                features = create_advanced_features(measured_labs_normalized)
            for i in range(3):
                # Only fit if not already fitted
                try:
                    # Check if model is already fitted by looking for 'coef_' attribute
                    getattr(best_models[i], 'coef_')
                except AttributeError:
                    best_models[i].fit(features, transformation_labs[:, i], sample_weight=best_weights)
        
        return best_models, best_degree, best_weights, best_delta_e
    
    def apply_multi_scale_correction(self, image_flat, models, degree, measured_labs_normalized):
        """
        Apply multi-scale polynomial correction to image data.
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        corrected_flat = np.zeros_like(image_flat)
        
        try:
            if degree == -1:  # Advanced features
                # Use advanced feature engineering
                advanced_features = self.create_advanced_features_for_prediction(image_flat)
                
                for i in range(3):
                    corrected_flat[:, i] = models[i].predict(advanced_features)
            
            elif degree == -2:  # Multi-scale
                # Use different degrees for different channels
                degrees = [1, 2, 2]  # L*, a*, b*
                
                for i, (model, deg) in enumerate(zip(models, degrees)):
                    if deg == 1:
                        features = image_flat
                    else:
                        poly = PolynomialFeatures(degree=deg, include_bias=True)
                        features = poly.fit_transform(image_flat)
                    
                    corrected_flat[:, i] = model.predict(features)
            
            elif degree == -3:  # Legacy color space optimization (removed)
                # Fallback to standard polynomial features
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, include_bias=True)
                features = poly.fit_transform(measured_labs_normalized)
            else:  # Standard polynomial
                if degree == 1:
                    features = image_flat
                else:
                    poly = PolynomialFeatures(degree=degree, include_bias=True)
                    features = poly.fit_transform(image_flat)
                
                for i in range(3):
                    corrected_flat[:, i] = models[i].predict(features)
            
            # Check for invalid predictions and replace with original values
            if np.any(np.isnan(corrected_flat)) or np.any(np.isinf(corrected_flat)):
                print("Warning: Invalid predictions detected, using original values")
                corrected_flat = np.where(
                    np.isnan(corrected_flat) | np.isinf(corrected_flat),
                    image_flat,
                    corrected_flat
                )
            
            return corrected_flat
            
        except Exception as e:
            print(f"Error in multi-scale correction: {e}")
            # Return original values if correction fails
            return image_flat
    
    def create_advanced_features_for_prediction(self, X):
        """Create advanced features for prediction (same as training)."""
        try:
            # Check for NaN values and replace them
            if np.any(np.isnan(X)):
                print("Warning: NaN values detected in prediction data, replacing with zeros")
                X = np.nan_to_num(X, nan=0.0)
            
            features = []
            
            # Basic polynomial features
            for degree in [1, 2, 3]:
                poly = PolynomialFeatures(degree=degree, include_bias=True)
                features.append(poly.fit_transform(X))
            
            # Interaction features
            L, a, b = X[:, 0], X[:, 1], X[:, 2]
            interactions = np.column_stack([
                L * a, L * b, a * b,  # 2-way interactions
                L * a * b,  # 3-way interaction
                L**2 * a, L**2 * b, a**2 * L, a**2 * b, b**2 * L, b**2 * a,  # Higher-order interactions
                np.sin(L * np.pi), np.cos(a * np.pi), np.sin(b * np.pi),  # Trigonometric features
                np.sqrt(np.abs(L)), np.sqrt(np.abs(a)), np.sqrt(np.abs(b))  # Root features
            ])
            features.append(interactions)
            
            result = np.hstack(features)
            
            # Check for NaN values in result and replace them
            if np.any(np.isnan(result)):
                print("Warning: NaN values detected in prediction feature matrix, replacing with zeros")
                result = np.nan_to_num(result, nan=0.0)
            
            return result
            
        except Exception as e:
            print(f"Error creating advanced features for prediction: {e}")
            # Fallback to simple polynomial features
            poly = PolynomialFeatures(degree=2, include_bias=True)
            return poly.fit_transform(X)

    def evaluate_correction_accuracy(self, measured_labs, reference_labs, corrected_labs):
        """
        Evaluate correction accuracy using professional metrics.
        Uses our custom CIEDE2000 implementation for accurate color difference calculation.
        """
        try:
            # Calculate ΔE for each patch using our custom CIEDE2000 implementation
            deltas = []
            for measured_lab, reference_lab, corrected_lab in zip(measured_labs, reference_labs, corrected_labs):
                # Calculate ΔE using our custom CIEDE2000 implementation
                delta_e = self.calculate_delta_e(corrected_lab, reference_lab, method='CIEDE2000')
                deltas.append(delta_e)
            
            # Calculate statistics
            avg_delta_e = np.mean(deltas)
            max_delta_e = np.max(deltas)
            min_delta_e = np.min(deltas)
            std_delta_e = np.std(deltas)
            
            # Quality assessment
            excellent_count = sum(1 for d in deltas if d < 2.0)
            good_count = sum(1 for d in deltas if 2.0 <= d < 4.0)
            acceptable_count = sum(1 for d in deltas if 4.0 <= d < 6.0)
            poor_count = sum(1 for d in deltas if d >= 6.0)
            
            print(f"\n=== Correction Accuracy Evaluation ===")
            print(f"Average ΔE: {avg_delta_e:.2f}")
            print(f"Maximum ΔE: {max_delta_e:.2f}")
            print(f"Minimum ΔE: {min_delta_e:.2f}")
            print(f"Standard Deviation: {std_delta_e:.2f}")
            print(f"\nQuality Distribution:")
            print(f"  Excellent (ΔE < 2.0): {excellent_count}/{len(deltas)} ({excellent_count/len(deltas)*100:.1f}%)")
            print(f"  Good (ΔE 2.0-4.0): {good_count}/{len(deltas)} ({good_count/len(deltas)*100:.1f}%)")
            print(f"  Acceptable (ΔE 4.0-6.0): {acceptable_count}/{len(deltas)} ({acceptable_count/len(deltas)*100:.1f}%)")
            print(f"  Poor (ΔE ≥ 6.0): {poor_count}/{len(deltas)} ({poor_count/len(deltas)*100:.1f}%)")
            
            return {
                'average_delta_e': avg_delta_e,
                'max_delta_e': max_delta_e,
                'min_delta_e': min_delta_e,
                'std_delta_e': std_delta_e,
                'deltas': deltas,
                'quality_distribution': {
                    'excellent': excellent_count,
                    'good': good_count,
                    'acceptable': acceptable_count,
                    'poor': poor_count
                }
            }
            
        except Exception as e:
            print(f"Accuracy evaluation failed: {e}")
            return None
    
    def apply_matrix_correction_improved(self, image_rgb, measured_labs, reference_labs):
        """
        Apply improved matrix correction with better bounds control.
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._get_correction_cache_key(measured_labs, reference_labs, "matrix")
            if hasattr(self, '_correction_cache') and cache_key in self._correction_cache:
                logger.info("Using cached matrix correction")
                cached_result = self._correction_cache[cache_key]
                return self._apply_cached_correction(image_rgb, cached_result)
            
            # Vectorized LAB to RGB conversion
            measured_rgbs = np.array([self.lab_to_rgb(lab) for lab in measured_labs], dtype=np.float32) / 255.0
            reference_rgbs = np.array([self.lab_to_rgb(lab) for lab in reference_labs], dtype=np.float32) / 255.0
            
            # Calculate transformation matrix using least squares with regularization
            regularization = 1e-6
            measured_rgbs_reg = measured_rgbs + np.random.normal(0, regularization, measured_rgbs.shape)
            matrix = np.linalg.pinv(measured_rgbs_reg) @ reference_rgbs
            
            # Apply matrix transformation (vectorized)
            image_rgb_norm = image_rgb.astype(np.float32) / 255.0
            h, w, _ = image_rgb_norm.shape
            image_reshaped = image_rgb_norm.reshape(-1, 3)
            
            # Vectorized matrix multiplication
            corrected_reshaped = image_reshaped @ matrix
            corrected_rgb = corrected_reshaped.reshape(h, w, 3)
            
            # Apply bounds checking and brightness preservation
            corrected_rgb = self._apply_brightness_preservation(image_rgb_norm, corrected_rgb)
            
            # Cache the result
            if not hasattr(self, '_correction_cache'):
                self._correction_cache = {}
            self._correction_cache[cache_key] = {
                'matrix': matrix,
                'method': 'matrix'
            }
            
            elapsed_time = time.time() - start_time
            logger.info(f"Matrix correction completed in {elapsed_time:.2f} seconds")
            
            return np.clip(corrected_rgb * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Matrix correction failed: {e}")
            return image_rgb
    
    def calculate_matrix_transformation(self, measured_labs, reference_labs):
        """Calculate the matrix transformation for LUT generation."""
        try:
            # Convert LAB values to RGB for matrix calculation
            measured_rgbs = []
            reference_rgbs = []
            
            for measured_lab, reference_lab in zip(measured_labs, reference_labs):
                measured_rgb = self.lab_to_rgb(measured_lab)
                reference_rgb = self.lab_to_rgb(reference_lab)
                
                measured_rgbs.append(measured_rgb)
                reference_rgbs.append(reference_rgb)
            
            measured_rgbs = np.array(measured_rgbs).astype(np.float32) / 255.0
            reference_rgbs = np.array(reference_rgbs).astype(np.float32) / 255.0
            
            # Calculate transformation matrix using least squares
            # Add a small regularization term to prevent overfitting
            regularization = 1e-6
            measured_rgbs_reg = measured_rgbs + np.random.normal(0, regularization, measured_rgbs.shape)
            
            # Solve for transformation matrix: reference = measured * matrix
            # Using pseudo-inverse for stability
            matrix = np.linalg.pinv(measured_rgbs_reg) @ reference_rgbs
            
            return matrix
            
        except Exception as e:
            print(f"Matrix transformation calculation failed: {e}")
            return None
    

    

    
    def test_reference_colors(self):
        """Create a test image showing all reference colors for comparison."""
        # Create a test image with all reference colors
        colors_per_row = 6
        patch_size = 60
        margin = 10
        
        # Calculate grid size
        num_colors = len(self.reference_colors_lab)
        num_rows = (num_colors + colors_per_row - 1) // colors_per_row
        
        # Create image
        img_width = colors_per_row * patch_size + (colors_per_row + 1) * margin
        img_height = num_rows * patch_size + (num_rows + 1) * margin + 100  # Extra space for labels
        
        test_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 128  # Gray background
        
        # Convert to PIL for drawing text
        from PIL import ImageDraw, ImageFont, Image
        test_pil = Image.fromarray(test_image)
        draw = ImageDraw.Draw(test_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Add colors and labels
        for i, (color_name, lab_values) in enumerate(self.reference_colors_lab.items()):
            row = i // colors_per_row
            col = i % colors_per_row
            
            # Calculate position
            x = margin + col * (patch_size + margin)
            y = margin + row * (patch_size + margin)
            
            # Get RGB color
            rgb_color = self.lab_to_rgb(lab_values)
            
            # Fill the patch
            test_pil.paste(Image.new('RGB', (patch_size, patch_size), tuple(rgb_color)), (x, y))
            
            # Draw label as white text with black outline
            label = color_name
            label_x = x + patch_size // 2
            label_y = y + patch_size + 5
            # Center the text
            bbox = draw.textbbox((0, 0), label, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            label_pos = (label_x - w // 2, label_y)
            draw.text(label_pos, label, font=font, fill="white", stroke_width=2, stroke_fill="black")
        
        # Create a new window to show the test image
        top = tk.Toplevel(self.root)
        top.title("Reference Color Test")
        top.geometry("800x600")
        
        # Create a canvas with scrollbars
        canvas = tk.Canvas(top)
        scrollbar_y = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(top, orient="horizontal", command=canvas.xview)
        
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Pack scrollbars and canvas
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Convert PIL image to PhotoImage
        test_photo = ImageTk.PhotoImage(test_pil)
        
        # Add image to canvas
        canvas.create_image(0, 0, anchor="nw", image=test_photo)
        canvas.image = test_photo  # Keep a reference
        
        # Configure canvas scrolling
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Add instructions
        instruction_label = ttk.Label(top, text="Compare these colors with your ColorChecker chart.\nIf they don't match, the reference values need updating.")
        instruction_label.pack(pady=10)
    


    def update_selected_color_display(self, color_name=None):
        # Use provided color name or find the last selected color to display
        if color_name is None:
            # Find the last selected color to display
            last_selected = None
            for color_name in self.selected_colors:
                if color_name not in [item.get() for item in self.color_listbox.curselection()]:
                    last_selected = color_name
                    break
            color_name = last_selected
        
        if not color_name or color_name not in self.selected_colors:
            return
        
        lab_color = self.selected_colors[color_name]
        ref_color = self.reference_colors_lab[color_name]
        
        # Convert LAB to RGB for preview
        rgb_color = self.lab_to_rgb(lab_color)
        
        # Create color preview
        color_preview = Image.new('RGB', (50, 20), tuple(rgb_color))
        color_preview_tk = ImageTk.PhotoImage(color_preview)
        
        # Update label
        self.selected_color_label.configure(
            text=f"{color_name}\nSelected: LAB{tuple(round(x,2) for x in lab_color)}\nReference: LAB{tuple(round(x,2) for x in ref_color)}",
            image=color_preview_tk
        )
        self.selected_color_label.image = color_preview_tk  # Keep reference
    


    def get_reference_colors_lab_cv(self):
        # Convert standard LAB to OpenCV LAB scale
        ref_cv = {}
        for k, (l, a, b) in self.reference_colors_lab.items():
            l_cv = l * 255.0 / 100.0
            a_cv = a + 128.0
            b_cv = b + 128.0
            ref_cv[k] = (l_cv, a_cv, b_cv)
        return ref_cv




    def apply_correction(self):
        if len(self.selected_colors) != len(self.reference_colors_lab):
            messagebox.showwarning("Warning", "Please select all 24 colors before applying correction.")
            return
        
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        
        try:
            # Set up D65 illuminant and reference ColorChecker
            D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
            REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
            
            # Get reference swatches in RGB
            REFERENCE_SWATCHES = colour.XYZ_to_RGB(
                colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
                "sRGB",
                REFERENCE_COLOUR_CHECKER.illuminant,
            )
            
            # Use original image directly (no preprocessing)
            wb_corrected_image = self.original_image_rgb
            
            # Step 2: Prepare measured swatches from selected colors
            self.progress_var.set("Preparing measured swatches...")
            self.progress_bar['value'] = 15
            self.root.update()
            
            # Convert selected LAB colors back to RGB for colour-science
            measured_swatches = []
            measured_labs = []
            reference_labs = []
            
            for color_name in self.reference_colors_lab.keys():
                if color_name in self.selected_colors:
                    lab_color = self.selected_colors[color_name]
                    measured_labs.append(lab_color)
                    reference_labs.append(self.reference_colors_lab[color_name])
                    # Convert LAB to RGB using skimage
                    rgb_color = self.lab_to_rgb(lab_color)
                    measured_swatches.append(rgb_color.astype(np.float32) / 255.0)
                else:
                    # Use reference if not selected
                    measured_swatches.append(REFERENCE_SWATCHES[list(self.reference_colors_lab.keys()).index(color_name)])
            

            
            measured_swatches = np.array(measured_swatches)
            
            # Apply colour correction
            self.progress_var.set("Applying colour correction...")
            self.progress_bar['value'] = 30
            self.root.update()
            
            # Apply correction based on selected method
            method = self.correction_method_var.get()
            
            if method == "Polynomial":
                # Use improved polynomial regression
                self.progress_var.set("Applying polynomial correction...")
                self.progress_bar['value'] = 35
                self.root.update()
                
                corrected_rgb = self.apply_polynomial_correction(wb_corrected_image, measured_labs, reference_labs)
                
                # Evaluate correction accuracy
                self.progress_var.set("Evaluating correction accuracy...")
                self.progress_bar['value'] = 40
                self.root.update()
                
                # Get corrected LAB values for evaluation
                corrected_labs = []
                for color_name in self.reference_colors_lab.keys():
                    if color_name == 'Black':  # Skip black patch
                        continue
                    if color_name in self.selected_rects:
                        orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                        region = corrected_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                        region_lab = rgb2lab(region.astype(np.float32) / 255.0)
                        avg_corrected_lab = np.mean(region_lab.reshape(-1, 3), axis=0)
                        corrected_labs.append(avg_corrected_lab)
                
                # Evaluate accuracy
                if len(corrected_labs) > 0:
                    evaluation_result = self.evaluate_correction_accuracy(
                        np.array(measured_labs), 
                        np.array(reference_labs), 
                        np.array(corrected_labs)
                    )
            else:  # Matrix
                # Use improved matrix correction with better bounds control
                self.progress_var.set("Applying improved matrix correction...")
                self.progress_bar['value'] = 35
                self.root.update()
                
                # Use our improved matrix correction method
                corrected_rgb = self.apply_matrix_correction_improved(wb_corrected_image, measured_labs, reference_labs)
                
                # Store matrix transformation for LUT generation
                self.current_matrix_transformation = self.calculate_matrix_transformation(measured_labs, reference_labs)
                
                # Convert back to 0-1 range for consistency
                corrected_rgb = corrected_rgb.astype(np.float32) / 255.0
            
            # Clip and convert back to uint8
            self.progress_bar['value'] = 90
            self.progress_var.set("Finalizing correction...")
            self.root.update()
            
            # Ensure corrected_rgb is in the right format
            if isinstance(corrected_rgb, np.ndarray) and corrected_rgb.dtype == np.uint8:
                self.corrected_image = corrected_rgb
            else:
                self.corrected_image = np.clip(corrected_rgb * 255, 0, 255).astype(np.uint8)
            
            # Update display
            self.progress_bar['value'] = 95
            self.progress_var.set("Updating display...")
            self.root.update()
            self.display_corrected_image()
            
            # Post-correction analysis
            self.progress_bar['value'] = 100
            self.progress_var.set("Analyzing results...")
            self.root.update()
            
            msg = "\nCorrected Patch Deltas (LAB):\n"
            for color_name, rect in self.selected_rects.items():
                if color_name == 'Black':  # Skip black patch
                    continue
                orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in rect]
                region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                region_lab = rgb2lab(region.astype(np.float32) / 255.0)
                avg_lab = np.mean(region_lab.reshape(-1, 3), axis=0)
                ref_lab_std = self.reference_colors_lab[color_name]
                delta = tuple(avg_lab[i] - ref_lab_std[i] for i in range(3))
                delta_2 = tuple(f"{x:.2f}" for x in delta)
                msg += f"{color_name}: {delta_2}\n"
            
            self.offset_text.config(state=tk.NORMAL)
            self.offset_text.insert(tk.END, msg)
            self.offset_text.see(tk.END)
            self.offset_text.config(state=tk.DISABLED)
            
            self.progress_var.set("Colour-science correction applied successfully!")
            self.progress_bar['value'] = 0  # Reset progress bar
            
        except Exception as e:
            self.progress_bar['value'] = 0  # Reset progress bar on error
            messagebox.showerror("Error", f"Failed to apply color correction: {str(e)}")
    
    def display_corrected_image(self):
        if self.corrected_image is None:
            return
        
        # Switch to showing corrected image and update button text
        self.showing_corrected = True
        if hasattr(self, 'before_after_button'):
            self.before_after_button.config(text="👁️ Show Before")
        
        # Use the existing display system with zoom/pan
        self.update_display_image()
    
    def save_image(self):
        if self.corrected_image is None:
            messagebox.showwarning("Warning", "No corrected image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Corrected Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert RGB back to BGR for OpenCV
                corrected_bgr = cv2.cvtColor(self.corrected_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, corrected_bgr)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def reset_selections(self):
        self.selected_colors = {}
        self.selected_rects = {} # Clear previous rectangles
        self.current_selection = None
        self.corrected_image = None
        self.showing_corrected = False  # Reset to show original image
        
        # Update before/after button text
        if hasattr(self, 'before_after_button'):
            self.before_after_button.config(text="👁️ Show After")
        
        if self.original_image_rgb is not None:
            # Redisplay original image
            self.update_display_image()
        
        self.selected_color_label.configure(text="No color selected", image="")
        self.progress_var.set("Ready")
        self.color_listbox.selection_clear(0, tk.END)

    def fill_dummy_data(self):
        """Fill self.selected_colors with dummy LAB values for testing and display offsets."""
        import random
        offsets = []
        self.selected_colors = {}
        self.selected_rects = {} # Clear previous rectangles
        for k, v in self.reference_colors_lab.items():
            # Add small random noise to simulate measurement error
            l, a, b = v
            l_ = l + random.uniform(-5, 5)
            a_ = a + random.uniform(-5, 5)
            b_ = b + random.uniform(-5, 5)
            # Store in standard LAB scale (not OpenCV scale)
            self.selected_colors[k] = (l_, a_, b_)
            # Calculate offset in standard LAB
            offsets.append((k, l_ - l, a_ - a, b_ - b))
        self.progress_var.set("Dummy data filled. Ready to apply correction.")
        self.update_color_list_display()
        # Display offsets in the text area (exclude black patch)
        msg = "LAB Offsets (Dummy - Reference):\n"
        avg_l = avg_a = avg_b = 0
        count = 0
        for k, dl, da, db in offsets:
            if k == 'Black':  # Skip black patch
                continue
            count += 1
            msg += f"{k}: dL={dl:+.2f}, da={da:+.2f}, db={db:+.2f}\n"
            avg_l += dl
            avg_a += da
            avg_b += db
        avg_l /= count
        avg_a /= count
        avg_b /= count
        msg += f"\nAverage offset: dL={avg_l:+.2f}, da={avg_a:+.2f}, db={avg_b:+.2f}"
        self.offset_text.config(state=tk.NORMAL)
        self.offset_text.delete(1.0, tk.END)
        self.offset_text.insert(tk.END, msg)
        self.offset_text.config(state=tk.DISABLED)

    def show_color_comparison(self):
        """Show a modern, redesigned color correction analysis screen."""
        if not self.selected_colors or self.corrected_image is None:
            messagebox.showwarning("Warning", "Please apply color correction first to see the comparison.")
            return
        
        # Create a new window for the comparison
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Color Correction Analysis")
        comparison_window.geometry("625x1000")
        
        # Create main container with modern styling
        main_container = ttk.Frame(comparison_window, padding="15")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        comparison_window.columnconfigure(0, weight=1)
        comparison_window.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Header section with modern design
        header_frame = ttk.Frame(main_container, style='Panel.TFrame')
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Title with modern styling
        title_label = ttk.Label(header_frame, text="Color Correction Analysis", style='Title.TLabel')
        title_label.pack(pady=20, padx=25)
        
        # Create content area with modern scrollable frame
        content_frame = ttk.Frame(main_container)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Create canvas and scrollbars with modern styling
        canvas = tk.Canvas(content_frame, highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(content_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        
        # Configure scrolling
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create scrollable frame
        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Calculate statistics first
        color_names = [name for name in self.selected_colors.keys() if name != 'Black']
        orig_deltas = []
        corr_deltas = []
        improvements = []
        
        for color_name in color_names:
            if color_name in self.selected_rects:
                orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                
                # Original delta
                orig_region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                orig_lab = rgb2lab(orig_region.astype(np.float32) / 255.0)
                avg_orig_lab = np.mean(orig_lab.reshape(-1, 3), axis=0)
                ref_lab = self.reference_colors_lab[color_name]
                orig_delta = np.array(avg_orig_lab) - np.array(ref_lab)
                orig_delta_mag = np.sqrt(np.sum(orig_delta**2))
                orig_deltas.append(orig_delta_mag)
                
                # Corrected delta
                corr_region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                corrected_lab = rgb2lab(corr_region.astype(np.float32) / 255.0)
                avg_corrected_lab = np.mean(corrected_lab.reshape(-1, 3), axis=0)
                corr_delta = np.array(avg_corrected_lab) - np.array(ref_lab)
                corr_delta_mag = np.sqrt(np.sum(corr_delta**2))
                corr_deltas.append(corr_delta_mag)
                
                # Improvement
                improvement = orig_delta_mag - corr_delta_mag
                improvements.append(improvement)
            # If not in selected_rects, skip
            else:
                pass
        
        # Create modern statistics dashboard
        if orig_deltas:
            avg_orig = np.mean(orig_deltas)
            avg_corr = np.mean(corr_deltas)
            avg_improvement = np.mean(improvements)
            
            # Quality distribution
            excellent_count = sum(1 for d in corr_deltas if d < 3)
            good_count = sum(1 for d in corr_deltas if 3 <= d < 6)
            needs_improvement_count = sum(1 for d in corr_deltas if d >= 6)
            
            # Statistics dashboard
            stats_frame = ttk.Frame(scrollable_frame, style='Panel.TFrame')
            stats_frame.pack(fill=tk.X, pady=(0, 20), padx=10)
            
            # Main stats row
            main_stats_frame = ttk.Frame(stats_frame)
            main_stats_frame.pack(fill=tk.X, pady=15)
            
            # Create stat cards
            stat_cards = [
                ("Original ΔE", f"{avg_orig:.2f}", "#e74c3c"),
                ("Corrected ΔE", f"{avg_corr:.2f}", "#27ae60"),
                ("Improvement", f"{avg_improvement:.2f}", "#f39c12")
            ]
            
            for i, (label, value, color) in enumerate(stat_cards):
                card_frame = ttk.Frame(main_stats_frame, style='Card.TFrame')
                card_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
                
                value_label = ttk.Label(card_frame, text=value, font=('Arial', 20, 'bold'), 
                                      foreground=color)
                value_label.pack(pady=(10, 5))
                
                label_label = ttk.Label(card_frame, text=label, style='Info.TLabel')
                label_label.pack(pady=(0, 10))
            
            # Quality distribution row
            quality_frame = ttk.Frame(stats_frame)
            quality_frame.pack(fill=tk.X, pady=(0, 15))
            
            quality_title = ttk.Label(quality_frame, text="Quality Distribution", style='Header.TLabel')
            quality_title.pack(pady=(0, 10))
            
            quality_cards_frame = ttk.Frame(quality_frame)
            quality_cards_frame.pack()
            
            quality_stats = [
                ("Excellent", f"{excellent_count}", "#27ae60", "ΔE < 3.0"),
                ("Good", f"{good_count}", "#f39c12", "ΔE 3.0-6.0"),
                ("Needs Work", f"{needs_improvement_count}", "#e74c3c", "ΔE > 6.0")
            ]
            
            for label, count, color, range_text in quality_stats:
                q_card = ttk.Frame(quality_cards_frame, style='Card.TFrame')
                q_card.pack(side=tk.LEFT, padx=10)
                
                count_label = ttk.Label(q_card, text=count, font=('Arial', 16, 'bold'), 
                                      foreground=color)
                count_label.pack(pady=(8, 2))
                
                label_label = ttk.Label(q_card, text=label, style='Header.TLabel')
                label_label.pack()
                
                range_label = ttk.Label(q_card, text=range_text, style='Info.TLabel')
                range_label.pack(pady=(0, 8))
        
        # Create modern color grid
        grid_frame = ttk.Frame(scrollable_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Configure grid weights
        cols = 4  # 4 columns for better layout
        for i in range(cols):
            grid_frame.columnconfigure(i, weight=1)
        
        # Create modern color comparison cards
        for i, color_name in enumerate(color_names):
            row = i // cols
            col = i % cols
            
            # Create modern card frame
            card_frame = ttk.Frame(grid_frame, style='Card.TFrame')
            card_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            
            # Card header with modern styling
            header_frame = ttk.Frame(card_frame, style='Panel.TFrame')
            header_frame.pack(fill=tk.X, pady=(0, 10))
            
            header_label = ttk.Label(header_frame, text=color_name, style='Header.TLabel')
            header_label.pack(pady=8)
            
            # Get colors
            ref_lab = self.reference_colors_lab[color_name]
            ref_rgb = self.lab_to_rgb(ref_lab)
            
            if color_name in self.selected_rects:
                orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                corrected_lab = rgb2lab(region.astype(np.float32) / 255.0)
                avg_corrected_lab = np.mean(corrected_lab.reshape(-1, 3), axis=0)
                corrected_rgb = self.lab_to_rgb(avg_corrected_lab)
            
                orig_region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                orig_lab = rgb2lab(orig_region.astype(np.float32) / 255.0)
                avg_orig_lab = np.mean(orig_lab.reshape(-1, 3), axis=0)
                orig_rgb = self.lab_to_rgb(avg_orig_lab)
                
                # Calculate deltas
                orig_delta = tuple(avg_orig_lab[i] - ref_lab[i] for i in range(3))
                orig_delta_magnitude = np.sqrt(sum(d**2 for d in orig_delta))
                corr_delta = tuple(avg_corrected_lab[i] - ref_lab[i] for i in range(3))
                corr_delta_magnitude = np.sqrt(sum(d**2 for d in corr_delta))
                improvement = orig_delta_magnitude - corr_delta_magnitude
                
                # Quality assessment
                if corr_delta_magnitude < 3:
                    quality_color = '#27ae60'
                    quality_text = "Excellent"
                elif corr_delta_magnitude < 6:
                    quality_color = '#f39c12'
                    quality_text = "Good"
                else:
                    quality_color = '#e74c3c'
                    quality_text = "Needs Work"
            else:
                orig_rgb = corrected_rgb = ref_rgb
                orig_delta_magnitude = corr_delta_magnitude = improvement = 0
                quality_color = '#cccccc'
                quality_text = "N/A"
            
            # Color swatches section
            swatches_frame = ttk.Frame(card_frame)
            swatches_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
            
            # Top row: Original and Corrected
            top_row = ttk.Frame(swatches_frame)
            top_row.pack(fill=tk.X, pady=(0, 8))
            
            # Original swatch
            orig_frame = ttk.Frame(top_row)
            orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
            
            orig_label = ttk.Label(orig_frame, text="Original", style='Info.TLabel')
            orig_label.pack(pady=(0, 3))
            
            orig_color_hex = f'#{orig_rgb[0]:02x}{orig_rgb[1]:02x}{orig_rgb[2]:02x}'
            orig_patch = tk.Frame(orig_frame, bg=orig_color_hex, relief=tk.FLAT, bd=0, height=40)
            orig_patch.pack(fill=tk.BOTH, expand=True)
            
            # Corrected swatch
            corr_frame = ttk.Frame(top_row)
            corr_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
            
            corr_label = ttk.Label(corr_frame, text="Corrected", style='Info.TLabel')
            corr_label.pack(pady=(0, 3))
            
            corr_color_hex = f'#{corrected_rgb[0]:02x}{corrected_rgb[1]:02x}{corrected_rgb[2]:02x}'
            corr_patch = tk.Frame(corr_frame, bg=corr_color_hex, relief=tk.FLAT, bd=0, height=40)
            corr_patch.pack(fill=tk.BOTH, expand=True)
            
            # Bottom row: Target
            bottom_row = ttk.Frame(swatches_frame)
            bottom_row.pack(fill=tk.X, pady=(8, 0))
            
            target_label = ttk.Label(bottom_row, text="Target", style='Info.TLabel')
            target_label.pack(pady=(0, 3))
            
            ref_color_hex = f'#{ref_rgb[0]:02x}{ref_rgb[1]:02x}{ref_rgb[2]:02x}'
            target_patch = tk.Frame(bottom_row, bg=ref_color_hex, relief=tk.FLAT, bd=0, height=25)
            target_patch.pack(fill=tk.X)
            
            # Metrics section
            metrics_frame = ttk.Frame(card_frame)
            metrics_frame.pack(fill=tk.X, padx=15, pady=10)
            
            # Quality indicator
            quality_frame = ttk.Frame(metrics_frame)
            quality_frame.pack(fill=tk.X, pady=(0, 5))
            
            quality_indicator = ttk.Label(quality_frame, text=quality_text, 
                                        font=('Arial', 10, 'bold'), foreground=quality_color)
            quality_indicator.pack()
            
            # Delta E values
            if color_name in self.selected_rects:
                delta_frame = ttk.Frame(metrics_frame)
                delta_frame.pack(fill=tk.X)
                
                # Original ΔE
                orig_delta_label = ttk.Label(delta_frame, text=f"Original: {orig_delta_magnitude:.2f}", 
                                           style='Info.TLabel')
                orig_delta_label.pack()
                
                # Corrected ΔE
                corr_delta_label = ttk.Label(delta_frame, text=f"Corrected: {corr_delta_magnitude:.2f}", 
                                           font=('Arial', 9, 'bold'), foreground=quality_color)
                corr_delta_label.pack()
                
                # Improvement
                if improvement > 0:
                    improvement_text = f"Improvement: +{improvement:.2f}"
                    improvement_color = '#27ae60'
                else:
                    improvement_text = f"Change: {improvement:.2f}"
                    improvement_color = '#e74c3c'
                
                improvement_label = ttk.Label(delta_frame, text=improvement_text, 
                                            style='Info.TLabel', foreground=improvement_color)
                improvement_label.pack()
        
        # Update scroll region
        scrollable_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def export_pdf_report(self):
        """Export a comprehensive color correction report to PDF."""
        if not self.selected_colors or self.corrected_image is None:
            messagebox.showwarning("Warning", "Please apply color correction first to generate a report.")
            return
        
        # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save Color Correction Report"
        )
        
        if not filename:
            return
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2c3e50')
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                textColor=colors.HexColor('#34495e')
            )
            
            header_style = ParagraphStyle(
                'CustomHeader',
                parent=styles['Heading3'],
                fontSize=14,
                spaceAfter=15,
                textColor=colors.HexColor('#2980b9')
            )
            
            # Title page
            story.append(Paragraph("Color Correction Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Report metadata
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata_text = f"""
            <b>Report Generated:</b> {current_time}<br/>
            <b>Correction Method:</b> {self.correction_method_var.get()}<br/>
            <b>Number of Color Patches:</b> {len(self.selected_colors)}<br/>
            <b>Image Dimensions:</b> {self.original_image_rgb.shape[1]} x {self.original_image_rgb.shape[0]} pixels
            """
            story.append(Paragraph(metadata_text, styles['Normal']))
            story.append(Spacer(1, 30))
            
            # Summary statistics
            story.append(Paragraph("Summary Statistics", subtitle_style))
            
            # Calculate statistics
            color_names = [name for name in self.selected_colors.keys() if name != 'Black']
            orig_deltas = []
            corr_deltas = []
            improvements = []
            
            for color_name in color_names:
                if color_name in self.selected_rects:
                    orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                    
                    # Original delta
                    orig_region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                    orig_lab = rgb2lab(orig_region.astype(np.float32) / 255.0)
                    avg_orig_lab = np.mean(orig_lab.reshape(-1, 3), axis=0)
                    ref_lab = self.reference_colors_lab[color_name]
                    orig_delta = np.array(avg_orig_lab) - np.array(ref_lab)
                    orig_delta_mag = np.sqrt(np.sum(orig_delta**2))
                    orig_deltas.append(orig_delta_mag)
                    
                    # Corrected delta
                    corr_region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                    corrected_lab = rgb2lab(corr_region.astype(np.float32) / 255.0)
                    avg_corrected_lab = np.mean(corrected_lab.reshape(-1, 3), axis=0)
                    corr_delta = np.array(avg_corrected_lab) - np.array(ref_lab)
                    corr_delta_mag = np.sqrt(np.sum(corr_delta**2))
                    corr_deltas.append(corr_delta_mag)
                    
                    # Improvement
                    improvement = orig_delta_mag - corr_delta_mag
                    improvements.append(improvement)
            
            if orig_deltas:
                avg_orig = np.mean(orig_deltas)
                avg_corr = np.mean(corr_deltas)
                avg_improvement = np.mean(improvements)
                max_improvement = max(improvements)
                min_improvement = min(improvements)
                
                # Quality distribution
                excellent_count = sum(1 for d in corr_deltas if d < 3)
                good_count = sum(1 for d in corr_deltas if 3 <= d < 6)
                needs_improvement_count = sum(1 for d in corr_deltas if d >= 6)
                
                stats_data = [
                    ['Metric', 'Value'],
                    ['Average Original ΔE', f'{avg_orig:.2f}'],
                    ['Average Corrected ΔE', f'{avg_corr:.2f}'],
                    ['Average Improvement', f'{avg_improvement:.2f}'],
                    ['Maximum Improvement', f'{max_improvement:.2f}'],
                    ['Minimum Improvement', f'{min_improvement:.2f}'],
                    ['', ''],
                    ['Quality Distribution', ''],
                    ['Excellent (ΔE < 3.0)', f'{excellent_count}/{len(corr_deltas)} ({excellent_count/len(corr_deltas)*100:.1f}%)'],
                    ['Good (ΔE 3.0-6.0)', f'{good_count}/{len(corr_deltas)} ({good_count/len(corr_deltas)*100:.1f}%)'],
                    ['Needs Improvement (ΔE ≥ 6.0)', f'{needs_improvement_count}/{len(corr_deltas)} ({needs_improvement_count/len(corr_deltas)*100:.1f}%)']
                ]
                
                stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                ]))
                
                story.append(stats_table)
                story.append(Spacer(1, 30))
            
            # Detailed color analysis
            story.append(Paragraph("Detailed Color Analysis", subtitle_style))
            
            # Create detailed table
            detailed_data = [['Color', 'Original ΔE', 'Corrected ΔE', 'Improvement', 'Quality']]
            
            for color_name in color_names:
                if color_name in self.selected_rects:
                    orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                    
                    # Original delta
                    orig_region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                    orig_lab = rgb2lab(orig_region.astype(np.float32) / 255.0)
                    avg_orig_lab = np.mean(orig_lab.reshape(-1, 3), axis=0)
                    ref_lab = self.reference_colors_lab[color_name]
                    orig_delta = np.array(avg_orig_lab) - np.array(ref_lab)
                    orig_delta_mag = np.sqrt(np.sum(orig_delta**2))
                    
                    # Corrected delta
                    corr_region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                    corrected_lab = rgb2lab(corr_region.astype(np.float32) / 255.0)
                    avg_corrected_lab = np.mean(corrected_lab.reshape(-1, 3), axis=0)
                    corr_delta = np.array(avg_corrected_lab) - np.array(ref_lab)
                    corr_delta_mag = np.sqrt(np.sum(corr_delta**2))
                    
                    # Improvement
                    improvement = orig_delta_mag - corr_delta_mag
                    
                    # Quality assessment
                    if corr_delta_mag < 3:
                        quality = "Excellent"
                        quality_color = colors.HexColor('#27ae60')
                    elif corr_delta_mag < 6:
                        quality = "Good"
                        quality_color = colors.HexColor('#f39c12')
                    else:
                        quality = "Needs Improvement"
                        quality_color = colors.HexColor('#e74c3c')
                    
                    detailed_data.append([
                        color_name,
                        f'{orig_delta_mag:.2f}',
                        f'{corr_delta_mag:.2f}',
                        f'{improvement:+.2f}',
                        quality
                    ])
            
            detailed_table = Table(detailed_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1.5*inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (1, 1), (3, -1), 'RIGHT'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (4, 1), (4, -1), 'CENTER'),
            ]))
            
            story.append(detailed_table)
            story.append(Spacer(1, 30))
            
            # Color visualization section
            story.append(Paragraph("Color Visualization", subtitle_style))
            
            # Create a function to generate color swatches
            def create_color_swatch(rgb_color, width=0.8, height=0.4):
                """Create a color swatch image for the PDF that fills the cell."""
                # Ensure RGB values are integers in valid range
                if isinstance(rgb_color, (list, np.ndarray)):
                    rgb_color = tuple(int(max(0, min(255, val))) for val in rgb_color)
                elif isinstance(rgb_color, tuple):
                    rgb_color = tuple(int(max(0, min(255, val))) for val in rgb_color)
                else:
                    # Fallback to a neutral gray if color is invalid
                    rgb_color = (128, 128, 128)
                
                # Create a PIL image with the color - make it larger for better quality
                img = Image.new('RGB', (200, 100), rgb_color)
                
                # Convert to bytes for ReportLab
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Create ReportLab image that fills the cell
                rl_img = RLImage(img_bytes, width=width*inch, height=height*inch)
                return rl_img
            
            # Create color comparison table
            color_data = [['Color Name', 'Original', 'Corrected', 'Target', 'Original ΔE', 'Corrected ΔE', 'Improvement']]
            
            for color_name in color_names:
                if color_name in self.selected_rects:
                    ref_lab = self.reference_colors_lab[color_name]
                    ref_rgb = self.lab_to_rgb(ref_lab)
                    
                    orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                    
                    # Original color
                    orig_region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                    orig_lab = rgb2lab(orig_region.astype(np.float32) / 255.0)
                    avg_orig_lab = np.mean(orig_lab.reshape(-1, 3), axis=0)
                    orig_rgb = self.lab_to_rgb(avg_orig_lab)
                    orig_delta = np.array(avg_orig_lab) - np.array(ref_lab)
                    orig_delta_mag = np.sqrt(np.sum(orig_delta**2))
                    
                    # Corrected color
                    corr_region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                    corrected_lab = rgb2lab(corr_region.astype(np.float32) / 255.0)
                    avg_corrected_lab = np.mean(corrected_lab.reshape(-1, 3), axis=0)
                    corrected_rgb = self.lab_to_rgb(avg_corrected_lab)
                    corr_delta = np.array(avg_corrected_lab) - np.array(ref_lab)
                    corr_delta_mag = np.sqrt(np.sum(corr_delta**2))
                    
                    # Debug logging for RGB values
                    logger.debug(f"{color_name} - Original RGB: {orig_rgb}, Corrected RGB: {corrected_rgb}, Target RGB: {ref_rgb}")
                    
                    # Improvement
                    improvement = orig_delta_mag - corr_delta_mag
                    
                    # Create color swatches with error handling
                    try:
                        orig_swatch = create_color_swatch(orig_rgb)
                        corr_swatch = create_color_swatch(corrected_rgb)
                        target_swatch = create_color_swatch(ref_rgb)
                    except Exception as e:
                        logger.warning(f"Failed to create color swatch for {color_name}: {e}")
                        # Create a fallback gray swatch
                        fallback_swatch = create_color_swatch((128, 128, 128))
                        orig_swatch = fallback_swatch
                        corr_swatch = fallback_swatch
                        target_swatch = fallback_swatch
                    
                    color_data.append([
                        color_name,
                        orig_swatch,
                        corr_swatch,
                        target_swatch,
                        f'{orig_delta_mag:.2f}',
                        f'{corr_delta_mag:.2f}',
                        f'{improvement:+.2f}'
                    ])
            
            # Create table with color swatches
            color_table = Table(color_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            color_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (4, 1), (6, -1), 'RIGHT'),
                ('VALIGN', (1, 1), (3, -1), 'MIDDLE'),
                # Remove padding from color swatch columns so they fill the cells
                ('LEFTPADDING', (1, 0), (1, -1), 0),  # Original column
                ('RIGHTPADDING', (1, 0), (1, -1), 0),
                ('LEFTPADDING', (2, 0), (2, -1), 0),  # Corrected column
                ('RIGHTPADDING', (2, 0), (2, -1), 0),
                ('LEFTPADDING', (3, 0), (3, -1), 0),  # Target column
                ('RIGHTPADDING', (3, 0), (3, -1), 0),
                ('TOPPADDING', (1, 0), (3, -1), 0),
                ('BOTTOMPADDING', (1, 0), (3, -1), 0),
            ]))
            
            story.append(color_table)
            story.append(Spacer(1, 20))
            
            # Add legend for color swatches
            legend_data = [
                ['Legend', ''],
                ['Original', 'Color as measured from the image'],
                ['Corrected', 'Color after applying correction algorithm'],
                ['Target', 'Reference color from ColorChecker standard']
            ]
            
            legend_table = Table(legend_data, colWidths=[1*inch, 4*inch])
            legend_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            
            story.append(legend_table)
            story.append(Spacer(1, 30))
            
            # LAB values section
            story.append(Paragraph("LAB Color Space Values", subtitle_style))
            
            lab_data = [['Color', 'Reference LAB', 'Original LAB', 'Corrected LAB']]
            
            for color_name in color_names:
                if color_name in self.selected_rects:
                    ref_lab = self.reference_colors_lab[color_name]
                    
                    orig_x0, orig_y0, orig_x1, orig_y1 = [int(v) for v in self.selected_rects[color_name]]
                    orig_region = self.original_image_rgb[orig_y0:orig_y1, orig_x0:orig_x1]
                    orig_lab = rgb2lab(orig_region.astype(np.float32) / 255.0)
                    avg_orig_lab = np.mean(orig_lab.reshape(-1, 3), axis=0)
                    
                    corr_region = self.corrected_image[orig_y0:orig_y1, orig_x0:orig_x1]
                    corrected_lab = rgb2lab(corr_region.astype(np.float32) / 255.0)
                    avg_corrected_lab = np.mean(corrected_lab.reshape(-1, 3), axis=0)
                    
                    lab_data.append([
                        color_name,
                        f'({ref_lab[0]:.1f}, {ref_lab[1]:.1f}, {ref_lab[2]:.1f})',
                        f'({avg_orig_lab[0]:.1f}, {avg_orig_lab[1]:.1f}, {avg_orig_lab[2]:.1f})',
                        f'({avg_corrected_lab[0]:.1f}, {avg_corrected_lab[1]:.1f}, {avg_corrected_lab[2]:.1f})'
                    ])
            
            lab_table = Table(lab_data, colWidths=[1.2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            lab_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ]))
            
            story.append(lab_table)
            story.append(Spacer(1, 30))
            
            # Recommendations section
            story.append(Paragraph("Recommendations", subtitle_style))
            
            recommendations = []
            if avg_corr > 6:
                recommendations.append("• Overall correction quality needs improvement. Consider using a different correction method or adjusting parameters.")
            
            if needs_improvement_count > len(corr_deltas) * 0.3:
                recommendations.append("• Many patches still need improvement. Consider using polynomial regression with higher degree.")
            
            if avg_improvement < 1:
                recommendations.append("• Limited improvement achieved. Check if the ColorChecker chart is properly lit and positioned.")
            
            if not recommendations:
                recommendations.append("• Excellent color correction achieved! The current settings are working well for this image.")
            
            recommendations.append("• For professional use, aim for ΔE values below 3.0 for all patches.")
            recommendations.append("• Consider the lighting conditions and ColorChecker chart quality for best results.")
            
            for rec in recommendations:
                story.append(Paragraph(rec, styles['Normal']))
                story.append(Spacer(1, 5))
            
            # Build PDF
            doc.build(story)
            
            messagebox.showinfo("Success", f"Color correction report saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create PDF report:\n{str(e)}")
            logger.error(f"PDF export error: {e}")



    def show_options_menu(self):
        """Show the options menu for processing and correction settings."""
        options_window = tk.Toplevel(self.root)
        options_window.title("Processing Options")
        options_window.geometry("600x800")  # Increased height for more content
        options_window.minsize(600, 600)
        options_window.resizable(True, True)
        
        # Add a canvas and scrollbar for scrollable content
        canvas = tk.Canvas(options_window, borderwidth=0, background="#f0f0f0")
        v_scrollbar = ttk.Scrollbar(options_window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create a frame inside the canvas
        main_frame = tk.Frame(canvas, bg="#f0f0f0")
        canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        main_frame.bind("<Configure>", on_frame_configure)
        
        # Title
        title_label = tk.Label(main_frame, text="Processing Options", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Correction Method Options
        method_frame = ttk.LabelFrame(main_frame, text="Correction Method", padding="10")
        method_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(method_frame, text="Select the color correction algorithm:").pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Radiobutton(method_frame, text="Matrix Correction (Recommended)", 
                       variable=self.correction_method_var, value="Matrix").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(method_frame, text="Polynomial Regression", 
                       variable=self.correction_method_var, value="Polynomial").pack(anchor=tk.W, pady=2)
        
        # Method descriptions
        desc_frame = tk.Frame(method_frame, bg='#ecf0f1', relief=tk.SUNKEN, bd=1)
        desc_frame.pack(fill=tk.X, pady=(10, 0))
        
        desc_text = """Matrix: Fast, stable, good for most cases
Polynomial: More accurate, but slower and may overfit"""
        
        desc_label = tk.Label(desc_frame, text=desc_text, font=('Arial', 9), 
                             bg='#ecf0f1', fg='#2c3e50', justify=tk.LEFT)
        desc_label.pack(padx=10, pady=10)
        
        # Advanced Options
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Polynomial-specific options
        poly_frame = ttk.Frame(advanced_frame)
        poly_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(poly_frame, text="Polynomial Options (when using Polynomial method):", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # Degree selection
        degree_frame = ttk.Frame(poly_frame)
        degree_frame.pack(fill=tk.X, pady=2)
        ttk.Label(degree_frame, text="Polynomial Degree:").pack(side=tk.LEFT)
        degree_combo = ttk.Combobox(degree_frame, textvariable=self.polynomial_degree_var, 
                                   values=["Auto", "1", "2", "3", "4"], state="readonly", width=10)
        degree_combo.pack(side=tk.RIGHT)
        
        # Advanced features checkboxes
        ttk.Checkbutton(poly_frame, text="Use Robust Regression (Huber/RANSAC)", 
                       variable=self.use_robust_regression).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(poly_frame, text="Use Cross-Validation", 
                       variable=self.use_cross_validation).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(poly_frame, text="Use Advanced Feature Engineering", 
                       variable=self.use_advanced_features).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(poly_frame, text="Use Multi-Scale Approach", 
                       variable=self.use_multi_scale).pack(anchor=tk.W, pady=1)

        
        # Weighting options
        ttk.Label(poly_frame, text="Patch Weighting Options:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        ttk.Checkbutton(poly_frame, text="Color-Only Weighting (exclude whites/greys/black)", 
                       variable=self.use_color_only_weighting).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(poly_frame, text="Greyscale Optimization (focus on whites/greys/black)", 
                       variable=self.use_greyscale_optimization).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(poly_frame, text="Skin Tone Priority (highest weight for skin tones)", 
                       variable=self.use_skin_tone_priority).pack(anchor=tk.W, pady=1)
        
        # General advanced options
        ttk.Label(advanced_frame, text="Other advanced options will be added here.").pack(anchor=tk.W, pady=(10, 0))
        
        # Performance Options
        performance_frame = ttk.LabelFrame(main_frame, text="Performance Options", padding="10")
        performance_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Performance level
        level_frame = ttk.Frame(performance_frame)
        level_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(level_frame, text="Optimization Level:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(level_frame, text="Fast (Speed Priority)", 
                       variable=self.optimization_level, value="Fast").pack(anchor=tk.W)
        ttk.Radiobutton(level_frame, text="Balanced (Speed + Quality)", 
                       variable=self.optimization_level, value="Balanced").pack(anchor=tk.W)
        ttk.Radiobutton(level_frame, text="Quality (Accuracy Priority)", 
                       variable=self.optimization_level, value="Quality").pack(anchor=tk.W)
        
        # Caching options
        cache_frame = ttk.Frame(performance_frame)
        cache_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(cache_frame, text="Caching & Memory:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Checkbutton(cache_frame, text="Enable correction caching", 
                       variable=self.enable_caching).pack(anchor=tk.W)
        ttk.Checkbutton(cache_frame, text="Use parallel processing", 
                       variable=self.use_parallel_processing).pack(anchor=tk.W)
        ttk.Checkbutton(cache_frame, text="Evaluate correction quality", 
                       variable=self.evaluate_quality).pack(anchor=tk.W)
        
        # LUT Processing Options
        lut_frame = ttk.Frame(performance_frame)
        lut_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(lut_frame, text="LUT Processing Method:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Radiobutton(lut_frame, text="Auto (Recommended)", 
                       variable=self.lut_processing_method, value="Auto").pack(anchor=tk.W)
        ttk.Radiobutton(lut_frame, text="GPU Acceleration (if available)", 
                       variable=self.lut_processing_method, value="GPU").pack(anchor=tk.W)
        ttk.Radiobutton(lut_frame, text="CPU Tiled Processing", 
                       variable=self.lut_processing_method, value="CPU_Tiled").pack(anchor=tk.W)
        ttk.Radiobutton(lut_frame, text="Standard CPU Processing", 
                       variable=self.lut_processing_method, value="CPU_Standard").pack(anchor=tk.W)
        
        # Performance stats
        stats_frame = ttk.Frame(performance_frame)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(stats_frame, text="System Information:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        stats = self.get_performance_stats()
        stats_text = f"""Cache Size: {stats['cache_size']} entries
Numba Available: {'Yes' if stats['numba_available'] else 'No'}
CuPy Available: {'Yes' if stats['cupy_available'] else 'No'}
GPU Memory: {'Available' if CUPY_AVAILABLE else 'Not Available'}"""
        
        stats_label = tk.Label(stats_frame, text=stats_text, font=('Consolas', 9), 
                              bg='white', fg='#2c3e50', justify=tk.LEFT, relief=tk.SUNKEN, bd=1)
        stats_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Current Settings Display
        settings_frame = ttk.LabelFrame(main_frame, text="Current Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Display current settings
        settings_text = f"""Correction Method: {self.correction_method_var.get()}
Polynomial Degree: {self.polynomial_degree_var.get()}
Robust Regression: {'Enabled' if self.use_robust_regression.get() else 'Disabled'}
Cross-Validation: {'Enabled' if self.use_cross_validation.get() else 'Disabled'}
Advanced Features: {'Enabled' if self.use_advanced_features.get() else 'Disabled'}
Multi-Scale: {'Enabled' if self.use_multi_scale.get() else 'Disabled'}
Color-Only Weighting: {'Enabled' if self.use_color_only_weighting.get() else 'Disabled'}
Greyscale Optimization: {'Enabled' if self.use_greyscale_optimization.get() else 'Disabled'}
Skin Tone Priority: {'Enabled' if self.use_skin_tone_priority.get() else 'Disabled'}

Performance Settings:
Optimization Level: {self.optimization_level.get()}
LUT Processing: {self.lut_processing_method.get()}
Caching: {'Enabled' if self.enable_caching.get() else 'Disabled'}
Parallel Processing: {'Enabled' if self.use_parallel_processing.get() else 'Disabled'}
Quality Evaluation: {'Enabled' if self.evaluate_quality.get() else 'Disabled'}"""
        
        settings_label = tk.Label(settings_frame, text=settings_text, font=('Arial', 10), 
                                 bg='white', fg='#2c3e50', justify=tk.LEFT, relief=tk.SUNKEN, bd=1)
        settings_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="Apply & Close", 
                  command=options_window.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", 
                  command=options_window.destroy).pack(side=tk.RIGHT)
        
        # Add utility buttons
        ttk.Button(button_frame, text="Clear Cache", 
                  command=self.clear_correction_cache).pack(side=tk.LEFT, padx=(0, 10))
        
        # Add a refresh button to update settings display
        def refresh_settings():
            settings_text = f"""Correction Method: {self.correction_method_var.get()}
Polynomial Degree: {self.polynomial_degree_var.get()}
Robust Regression: {'Enabled' if self.use_robust_regression.get() else 'Disabled'}
Cross-Validation: {'Enabled' if self.use_cross_validation.get() else 'Disabled'}
Advanced Features: {'Enabled' if self.use_advanced_features.get() else 'Disabled'}
Multi-Scale: {'Enabled' if self.use_multi_scale.get() else 'Disabled'}
Color-Only Weighting: {'Enabled' if self.use_color_only_weighting.get() else 'Disabled'}
Greyscale Optimization: {'Enabled' if self.use_greyscale_optimization.get() else 'Disabled'}
Skin Tone Priority: {'Enabled' if self.use_skin_tone_priority.get() else 'Disabled'}

Performance Settings:
Optimization Level: {self.optimization_level.get()}
LUT Processing: {self.lut_processing_method.get()}
Caching: {'Enabled' if self.enable_caching.get() else 'Disabled'}
Parallel Processing: {'Enabled' if self.use_parallel_processing.get() else 'Disabled'}
Quality Evaluation: {'Enabled' if self.evaluate_quality.get() else 'Disabled'}"""
            settings_label.config(text=settings_text)
        
        ttk.Button(button_frame, text="Refresh", 
                  command=refresh_settings).pack(side=tk.LEFT)

    def calculate_delta_e(self, lab1, lab2, method='CIEDE2000'):
        """
        Calculate color difference using industry-standard methods.
        
        Args:
            lab1: First LAB color (L*, a*, b*)
            lab2: Second LAB color (L*, a*, b*)
            method: 'CIEDE76', 'CIEDE94', or 'CIEDE2000' (default)
        
        Returns:
            ΔE value according to the specified method
        """
        try:
            if method == 'CIEDE76':
                # Simple Euclidean distance in LAB space
                return np.sqrt(np.sum((lab1 - lab2) ** 2))
            
            elif method == 'CIEDE94':
                # CIEDE94 calculation
                L1, a1, b1 = lab1
                L2, a2, b2 = lab2
                
                dL = L1 - L2
                C1 = np.sqrt(a1**2 + b1**2)
                C2 = np.sqrt(a2**2 + b2**2)
                dC = C1 - C2
                
                da = a1 - a2
                db = b1 - b2
                
                # Calculate dH
                dH_squared = da**2 + db**2 - dC**2
                dH = np.sqrt(max(0, dH_squared))
                
                # Weighting factors
                kL, kC, kH = 1.0, 1.0, 1.0
                K1, K2 = 0.045, 0.015
                
                SL = 1.0
                SC = 1.0 + K1 * C1
                SH = 1.0 + K2 * C1
                
                # Calculate ΔE94
                dE94 = np.sqrt((dL/(kL*SL))**2 + (dC/(kC*SC))**2 + (dH/(kH*SH))**2)
                return dE94
            
            elif method == 'CIEDE2000':
                # CIEDE2000 calculation (most accurate)
                L1, a1, b1 = lab1
                L2, a2, b2 = lab2
                
                # Step 1: Calculate C1, C2, Cb
                C1 = np.sqrt(a1**2 + b1**2)
                C2 = np.sqrt(a2**2 + b2**2)
                Cb = (C1 + C2) / 2
                
                # Step 2: Calculate G
                G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25**7)))
                
                # Step 3: Calculate a1', a2'
                a1_prime = a1 * (1 + G)
                a2_prime = a2 * (1 + G)
                
                # Step 4: Calculate C1', C2', Cb'
                C1_prime = np.sqrt(a1_prime**2 + b1**2)
                C2_prime = np.sqrt(a2_prime**2 + b2**2)
                Cb_prime = (C1_prime + C2_prime) / 2
                
                # Step 5: Calculate h1', h2'
                h1_prime = np.arctan2(b1, a1_prime)
                h2_prime = np.arctan2(b2, a2_prime)
                
                # Convert to degrees
                h1_prime = np.degrees(h1_prime)
                h2_prime = np.degrees(h2_prime)
                
                # Step 6: Calculate ΔL', ΔC', ΔH'
                dL_prime = L2 - L1
                dC_prime = C2_prime - C1_prime
                
                # Calculate ΔH'
                dh_prime = h2_prime - h1_prime
                if dh_prime > 180:
                    dh_prime -= 360
                elif dh_prime < -180:
                    dh_prime += 360
                
                dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
                
                # Step 7: Calculate H'
                H_prime = h1_prime + h2_prime
                if abs(h1_prime - h2_prime) > 180:
                    H_prime += 360
                H_prime /= 2
                
                # Step 8: Calculate T
                T = 1 - 0.17 * np.cos(np.radians(H_prime - 30)) + 0.24 * np.cos(np.radians(2 * H_prime)) + \
                    0.32 * np.cos(np.radians(3 * H_prime + 6)) - 0.20 * np.cos(np.radians(4 * H_prime - 63))
                
                # Step 9: Calculate SL, SC, SH, RT
                SL = 1 + (0.015 * (L1 + L2 - 50)**2) / np.sqrt(20 + (L1 + L2 - 50)**2)
                SC = 1 + 0.045 * Cb_prime
                SH = 1 + 0.015 * Cb_prime * T
                
                # Calculate RT
                RT = -2 * np.sqrt(Cb_prime**7 / (Cb_prime**7 + 25**7)) * np.sin(np.radians(60 * np.exp(-((H_prime - 275) / 25)**2)))
                
                # Step 10: Calculate ΔE00
                dE00 = np.sqrt((dL_prime / SL)**2 + (dC_prime / SC)**2 + (dH_prime / SH)**2 + RT * (dC_prime / SC) * (dH_prime / SH))
                
                return dE00
            
            else:
                raise ValueError(f"Unknown ΔE method: {method}")
                
        except Exception as e:
            print(f"Error calculating ΔE ({method}): {e}")
            # Fallback to simple Euclidean distance
            return np.sqrt(np.sum((lab1 - lab2) ** 2))
    
    def evaluate_polynomial_quality(self, measured_labs, reference_labs, corrected_labs, models, degree, delta_e_method='CIEDE2000'):
        """
        Comprehensive evaluation of polynomial correction quality.
        Provides detailed statistics and recommendations using industry-standard ΔE calculations.
        
        Args:
            measured_labs: Measured LAB values
            reference_labs: Reference LAB values  
            corrected_labs: Corrected LAB values
            models: Fitted polynomial models
            degree: Polynomial degree used
            delta_e_method: ΔE calculation method ('CIEDE76', 'CIEDE94', 'CIEDE2000')
        """
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        print("\n" + "="*60)
        print(f"POLYNOMIAL CORRECTION QUALITY EVALUATION ({delta_e_method})")
        print("="*60)
        
        # Calculate per-patch ΔE values using industry-standard method
        delta_e_values = []
        patch_analysis = []
        
        for i, (measured, reference, corrected) in enumerate(zip(measured_labs, reference_labs, corrected_labs)):
            # Calculate ΔE using the specified method
            try:
                # Ensure we have valid 3D LAB values
                if len(corrected) == 3 and len(reference) == 3:
                    delta_e = self.calculate_delta_e(corrected, reference, method=delta_e_method)
                else:
                    # Fallback to simple Euclidean distance
                    delta_e = np.sqrt(np.sum((corrected - reference) ** 2))
                delta_e_values.append(delta_e)
            except Exception as e:
                print(f"Error calculating ΔE for patch {i}: {e}")
                # Fallback to simple Euclidean distance
                delta_e = np.sqrt(np.sum((corrected - reference) ** 2))
                delta_e_values.append(delta_e)
            
            # Determine quality category based on industry standards
            if delta_e_method == 'CIEDE2000':
                # CIEDE2000 thresholds (most stringent)
                if delta_e < 1.0:
                    quality = "Excellent"
                    quality_color = "green"
                elif delta_e < 2.0:
                    quality = "Good"
                    quality_color = "orange"
                elif delta_e < 3.0:
                    quality = "Acceptable"
                    quality_color = "yellow"
                else:
                    quality = "Needs Improvement"
                    quality_color = "red"
            elif delta_e_method == 'CIEDE94':
                # CIEDE94 thresholds
                if delta_e < 1.5:
                    quality = "Excellent"
                    quality_color = "green"
                elif delta_e < 3.0:
                    quality = "Good"
                    quality_color = "orange"
                elif delta_e < 4.5:
                    quality = "Acceptable"
                    quality_color = "yellow"
                else:
                    quality = "Needs Improvement"
                    quality_color = "red"
            else:  # CIEDE76
                # CIEDE76 thresholds (least stringent)
                if delta_e < 2.3:
                    quality = "Excellent"
                    quality_color = "green"
                elif delta_e < 4.6:
                    quality = "Good"
                    quality_color = "orange"
                elif delta_e < 6.9:
                    quality = "Acceptable"
                    quality_color = "yellow"
                else:
                    quality = "Needs Improvement"
                    quality_color = "red"
            
            patch_analysis.append({
                'patch_id': i,
                'delta_e': delta_e,
                'quality': quality,
                'quality_color': quality_color,
                'measured': measured,
                'reference': reference,
                'corrected': corrected
            })
        
        # Overall statistics
        avg_delta_e = np.mean(delta_e_values)
        std_delta_e = np.std(delta_e_values)
        max_delta_e = np.max(delta_e_values)
        min_delta_e = np.min(delta_e_values)
        
        # Quality distribution based on method
        if delta_e_method == 'CIEDE2000':
            excellent_count = sum(1 for de in delta_e_values if de < 1.0)
            good_count = sum(1 for de in delta_e_values if 1.0 <= de < 2.0)
            acceptable_count = sum(1 for de in delta_e_values if 2.0 <= de < 3.0)
            poor_count = sum(1 for de in delta_e_values if de >= 3.0)
        elif delta_e_method == 'CIEDE94':
            excellent_count = sum(1 for de in delta_e_values if de < 1.5)
            good_count = sum(1 for de in delta_e_values if 1.5 <= de < 3.0)
            acceptable_count = sum(1 for de in delta_e_values if 3.0 <= de < 4.5)
            poor_count = sum(1 for de in delta_e_values if de >= 4.5)
        else:  # CIEDE76
            excellent_count = sum(1 for de in delta_e_values if de < 2.3)
            good_count = sum(1 for de in delta_e_values if 2.3 <= de < 4.6)
            acceptable_count = sum(1 for de in delta_e_values if 4.6 <= de < 6.9)
            poor_count = sum(1 for de in delta_e_values if de >= 6.9)
        
        # Model performance metrics
        model_performance = []
        for i, model in enumerate(models):
            channel_names = ['L*', 'a*', 'b*']
            
            # Calculate R² score for each channel
            measured_channel = measured_labs[:, i]
            reference_channel = reference_labs[:, i]
            
            # Predict using the model
            if degree == -1:  # Advanced features
                features = self.create_advanced_features_for_prediction(measured_labs)
            elif degree == -2:  # Multi-scale
                if i == 0:  # L* channel
                    features = measured_labs
                else:  # a* and b* channels
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=2, include_bias=True)
                    features = poly.fit_transform(measured_labs)
            elif degree == -3:  # Legacy color space optimization (removed)
                # Fallback to standard polynomial features
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, include_bias=True)
                features = poly.fit_transform(measured_labs)
            else:  # Standard polynomial
                if degree == 1:
                    features = measured_labs
                else:
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=degree, include_bias=True)
                    features = poly.fit_transform(measured_labs)
            
            predicted_channel = model.predict(features)
            r2 = r2_score(reference_channel, predicted_channel)
            mse = mean_squared_error(reference_channel, predicted_channel)
            
            model_performance.append({
                'channel': channel_names[i],
                'r2_score': r2,
                'mse': mse,
                'correlation': np.corrcoef(reference_channel, predicted_channel)[0, 1]
            })
        
        # Print comprehensive report
        print(f"\nOVERALL PERFORMANCE ({delta_e_method}):")
        print(f"  Average ΔE: {avg_delta_e:.3f} ± {std_delta_e:.3f}")
        print(f"  Best ΔE: {min_delta_e:.3f}")
        print(f"  Worst ΔE: {max_delta_e:.3f}")
        print(f"  Polynomial Degree: {degree}")
        
        print(f"\nQUALITY DISTRIBUTION ({delta_e_method}):")
        print(f"  Excellent: {excellent_count}/{len(delta_e_values)} ({excellent_count/len(delta_e_values)*100:.1f}%)")
        print(f"  Good: {good_count}/{len(delta_e_values)} ({good_count/len(delta_e_values)*100:.1f}%)")
        print(f"  Acceptable: {acceptable_count}/{len(delta_e_values)} ({acceptable_count/len(delta_e_values)*100:.1f}%)")
        print(f"  Needs Improvement: {poor_count}/{len(delta_e_values)} ({poor_count/len(delta_e_values)*100:.1f}%)")
        
        print(f"\nMODEL PERFORMANCE BY CHANNEL:")
        for perf in model_performance:
            print(f"  {perf['channel']}: R² = {perf['r2_score']:.3f}, MSE = {perf['mse']:.4f}, Corr = {perf['correlation']:.3f}")
        
        # Detailed patch analysis
        print(f"\nDETAILED PATCH ANALYSIS:")
        print(f"{'Patch':<6} {'ΔE':<10} {'Quality':<15} {'Measured LAB':<25} {'Reference LAB':<25} {'Corrected LAB':<25}")
        print("-" * 125)
        
        for analysis in patch_analysis:
            try:
                measured_str = f"({analysis['measured'][0]:.1f}, {analysis['measured'][1]:.1f}, {analysis['measured'][2]:.1f})"
                reference_str = f"({analysis['reference'][0]:.1f}, {analysis['reference'][1]:.1f}, {analysis['reference'][2]:.1f})"
                corrected_str = f"({analysis['corrected'][0]:.1f}, {analysis['corrected'][1]:.1f}, {analysis['corrected'][2]:.1f})"
                
                print(f"{analysis['patch_id']:<6} {analysis['delta_e']:<10.3f} {analysis['quality']:<15} {measured_str:<25} {reference_str:<25} {corrected_str:<25}")
            except Exception as e:
                print(f"{analysis['patch_id']:<6} {analysis['delta_e']:<10.3f} {analysis['quality']:<15} {'Error formatting':<25} {'Error formatting':<25} {'Error formatting':<25}")
        
        # Recommendations based on industry standards
        print(f"\nRECOMMENDATIONS ({delta_e_method}):")
        
        if delta_e_method == 'CIEDE2000':
            if avg_delta_e < 1.0:
                print("  ✓ Excellent overall correction quality! Meets professional standards.")
            elif avg_delta_e < 2.0:
                print("  ✓ Good correction quality. Suitable for most professional applications.")
            elif avg_delta_e < 3.0:
                print("  ⚠ Acceptable quality. Consider fine-tuning for critical applications.")
            else:
                print("  ⚠ Quality needs improvement for professional use. Consider:")
                print("    - Using a higher polynomial degree")
                print("    - Enabling advanced features")
                print("    - Using robust regression")
                print("    - Checking for outlier patches")
        else:
            if avg_delta_e < 2.0:
                print("  ✓ Excellent overall correction quality!")
            elif avg_delta_e < 4.0:
                print("  ✓ Good correction quality. Consider fine-tuning for specific patches.")
            else:
                print("  ⚠ Correction quality needs improvement. Consider:")
                print("    - Using a higher polynomial degree")
                print("    - Enabling advanced features")
                print("    - Using robust regression")
                print("    - Checking for outlier patches")
        
        # Channel-specific recommendations
        for perf in model_performance:
            if perf['r2_score'] < 0.8:
                print(f"  ⚠ {perf['channel']} channel has low R² ({perf['r2_score']:.3f}). Consider:")
                print(f"    - Using different polynomial degree for this channel")
                print(f"    - Enabling multi-scale approach")
        
        # Identify problematic patches
        if delta_e_method == 'CIEDE2000':
            problematic_patches = [p for p in patch_analysis if p['delta_e'] > 3.0]
        elif delta_e_method == 'CIEDE94':
            problematic_patches = [p for p in patch_analysis if p['delta_e'] > 4.5]
        else:
            problematic_patches = [p for p in patch_analysis if p['delta_e'] > 6.9]
            
        if problematic_patches:
            print(f"\nPROBLEMATIC PATCHES (ΔE > threshold):")
            for patch in problematic_patches:
                print(f"  Patch {patch['patch_id']}: ΔE = {patch['delta_e']:.3f}")
                print(f"    Measured: {patch['measured']}")
                print(f"    Reference: {patch['reference']}")
                print(f"    Corrected: {patch['corrected']}")
        
        print("\n" + "="*60)
        
        return {
            'overall_stats': {
                'avg_delta_e': avg_delta_e,
                'std_delta_e': std_delta_e,
                'max_delta_e': max_delta_e,
                'min_delta_e': min_delta_e
            },
            'quality_distribution': {
                'excellent': excellent_count,
                'good': good_count,
                'acceptable': acceptable_count,
                'poor': poor_count,
                'total': len(delta_e_values)
            },
            'model_performance': model_performance,
            'patch_analysis': patch_analysis,
            'problematic_patches': problematic_patches,
            'delta_e_method': delta_e_method
        }

    @staticmethod
    def is_greyscale_patch(L, a, b, l_thresh_high=85, l_thresh_low=15, sat_thresh=6.5):
        """
        Utility to determine if a patch is greyscale (white, grey, black) based on L*, a*, b*.
        Args:
            L: L* value (0-100)
            a: a* value
            b: b* value
            l_thresh_high: L* threshold for white
            l_thresh_low: L* threshold for black
            sat_thresh: saturation threshold for neutral
        Returns:
            True if patch is greyscale, False otherwise
        """
        saturation = np.sqrt(a**2 + b**2)
        return (saturation < sat_thresh) or (L > l_thresh_high) or (L < l_thresh_low)

    def export_icc_profile(self):
        """Export an ICC profile based on the current polynomial color correction."""
        if not self.selected_colors or self.corrected_image is None:
            messagebox.showwarning("Warning", "Please apply color correction first to generate an ICC profile.")
            return
        
        if self.correction_method_var.get() != "Polynomial":
            messagebox.showwarning("Warning", "ICC profile export is currently only supported for polynomial correction.")
            return
        
        # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".icc",
            filetypes=[("ICC Profile files", "*.icc"), ("All files", "*.*")],
            title="Save ICC Profile"
        )
        
        if not filename:
            return
        
        try:
            # Generate ICC profile from polynomial correction
            self.generate_icc_profile_from_polynomial(filename)
            messagebox.showinfo("Success", f"ICC profile saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create ICC profile:\n{str(e)}")
            logger.error(f"ICC export error: {e}")
    
    def generate_icc_profile_from_polynomial(self, filename):
        """Generate an ICC profile from polynomial correction models."""
        try:
            # Check if we have stored polynomial models from the last correction
            if self.current_polynomial_models is None or self.current_polynomial_degree is None:
                raise ValueError("No polynomial models available. Please apply polynomial correction first.")
            
            # Check if we have the transformer for degree > 1
            if self.current_polynomial_degree > 1 and self.current_polynomial_transformer is None:
                raise ValueError("Polynomial transformer not available. Please apply polynomial correction first.")
            
            # Use the stored models from the last correction
            best_models = self.current_polynomial_models
            best_degree = self.current_polynomial_degree
            
            # Create a 3D LUT for the ICC profile
            lut_size = 33  # Standard ICC LUT size
            lut = self.create_3d_lut_from_polynomial(best_models, best_degree, lut_size)
            
            # Create ICC profile using colour-science
            self.create_icc_profile_with_lut(lut, filename, best_degree)
            
        except Exception as e:
            logger.error(f"Error generating ICC profile: {e}")
            raise
    
    def create_3d_lut_from_polynomial(self, models, degree, lut_size=33):
        """Create a 3D LUT from polynomial models."""
        from sklearn.preprocessing import PolynomialFeatures
        from skimage.color import rgb2lab, lab2rgb
        
        logger.info(f"create_3d_lut_from_polynomial called with models: {models}, degree: {degree}, lut_size: {lut_size}")
        
        if models is None or degree is None:
            logger.warning("Models or degree is None, returning None")
            return None
            
        logger.info(f"Models type: {type(models)}, length: {len(models) if models else 'N/A'}")
        logger.info(f"Degree: {degree}")
        
        # Create RGB grid
        r_values = np.linspace(0, 1, lut_size)
        g_values = np.linspace(0, 1, lut_size)
        b_values = np.linspace(0, 1, lut_size)
        
        # Create full 3D grid
        R, G, B = np.meshgrid(r_values, g_values, b_values, indexing='ij')
        
        # Reshape to 2D array for processing
        rgb_grid = np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)
        logger.info(f"RGB grid shape: {rgb_grid.shape}")
        
        # Convert RGB to LAB (same as polynomial correction)
        lab_grid = rgb2lab(rgb_grid)
        
        # Normalize LAB values (same as polynomial correction)
        lab_normalized = lab_grid.copy()
        lab_normalized[:, 0] = lab_normalized[:, 0] / 100.0  # L*
        lab_normalized[:, 1:] = (lab_normalized[:, 1:] + 128.0) / 255.0  # a*, b*
        
        logger.info(f"LAB normalized grid shape: {lab_normalized.shape}")
        
        # Apply polynomial transformation to LAB values
        if degree == -1:  # Advanced features
            logger.info("Applying advanced feature transformation")
            # Use the stored polynomial transformer if available, otherwise create advanced features
            if hasattr(self, 'current_polynomial_transformer') and self.current_polynomial_transformer is not None:
                logger.info("Using stored polynomial transformer")
                features = self.current_polynomial_transformer.transform(lab_normalized)
            else:
                logger.info("Creating advanced features for prediction")
                features = self.create_advanced_features_for_prediction(lab_normalized)
            
            # Apply each channel's model to predict transformation
            predicted_transformation = np.zeros_like(lab_normalized)
            for i, model in enumerate(models):
                logger.info(f"Applying advanced model {i}: {type(model)}")
                predicted_transformation[:, i] = model.predict(features)
            
            # Add transformation to input (same as polynomial correction)
            corrected_lab_normalized = lab_normalized + predicted_transformation
                
        elif degree == -2:  # Multi-scale
            logger.info("Applying multi-scale transformation")
            # Use different degrees for different channels
            degrees = [1, 2, 2]  # L*, a*, b*
            
            predicted_transformation = np.zeros_like(lab_normalized)
            for i, (model, deg) in enumerate(zip(models, degrees)):
                if deg == 1:
                    features = lab_normalized
                else:
                    poly = PolynomialFeatures(degree=deg, include_bias=True)
                    features = poly.fit_transform(lab_normalized)
                
                logger.info(f"Applying multi-scale model {i} (degree {deg}): {type(model)}")
                predicted_transformation[:, i] = model.predict(features)
            
            # Add transformation to input
            corrected_lab_normalized = lab_normalized + predicted_transformation
                
        elif degree > 0:  # Standard polynomial
            logger.info(f"Applying polynomial transformation with degree {degree}")
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            lab_poly = poly.fit_transform(lab_normalized)
            logger.info(f"Polynomial features shape: {lab_poly.shape}")
            
            # Apply each channel's polynomial model to predict transformation
            predicted_transformation = np.zeros_like(lab_normalized)
            for i, model in enumerate(models):
                logger.info(f"Applying model {i}: {type(model)}")
                predicted_transformation[:, i] = model.predict(lab_poly)
            
            # Add transformation to input
            corrected_lab_normalized = lab_normalized + predicted_transformation
        else:
            # For degree 0 or other negative values, use identity transformation
            logger.info("Using identity transformation")
            corrected_lab_normalized = lab_normalized
        
        # Denormalize corrected LAB values (same as polynomial correction)
        corrected_lab = corrected_lab_normalized.copy()
        corrected_lab[:, 0] = corrected_lab[:, 0] * 100.0  # L*
        corrected_lab[:, 1:] = corrected_lab[:, 1:] * 255.0 - 128.0  # a*, b*
        
        # Clip to valid LAB ranges
        corrected_lab[:, 0] = np.clip(corrected_lab[:, 0], 0, 100)  # L*
        corrected_lab[:, 1:] = np.clip(corrected_lab[:, 1:], -128, 127)  # a*, b*
        
        # Convert back to RGB
        corrected_rgb = lab2rgb(corrected_lab)
        corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
        
        # Clip to valid range
        corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
        
        # Reshape back to 3D LUT
        lut = corrected_rgb.reshape(lut_size, lut_size, lut_size, 3)
        logger.info(f"Final LUT shape: {lut.shape}")
        
        return lut

    def create_3d_lut_from_matrix(self, matrix, lut_size=33):
        """Create a 3D LUT from matrix transformation."""
        # Create RGB grid
        r_values = np.linspace(0, 1, lut_size)
        g_values = np.linspace(0, 1, lut_size)
        b_values = np.linspace(0, 1, lut_size)
        
        # Create full 3D grid
        R, G, B = np.meshgrid(r_values, g_values, b_values, indexing='ij')
        
        # Reshape to 2D array for processing
        rgb_grid = np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)
        
        # Apply matrix transformation directly
        corrected_rgb = rgb_grid @ matrix
        
        # Clip to valid range
        corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
        
        # Reshape back to 3D LUT
        lut = corrected_rgb.reshape(lut_size, lut_size, lut_size, 3)
        
        return lut
    
    def create_icc_profile_with_lut(self, lut, filename, degree):
        """Create and save ICC profile using colour-science."""
        try:
            # Create a basic ICC profile structure
            profile_name = f"ColorChecker_Polynomial_Degree_{degree}"
            
            # Create ICC profile using colour-science
            profile = colour.ICC_Profile()
            
            # Set basic profile information
            profile.profile_description = profile_name
            profile.copyright = "Generated by ColorChecker Tool"
            profile.model = "ColorChecker Polynomial Correction"
            
            # Create A2B0 tag (RGB to PCS transformation)
            # This represents our polynomial correction
            a2b0 = colour.ICC_Profile_Tag()
            a2b0.signature = b'A2B0'
            
            # Create LUT A2B0 tag
            lut_a2b0 = colour.ICC_Profile_Tag_LUT_A2B0()
            lut_a2b0.input_channels = 3  # RGB
            lut_a2b0.output_channels = 3  # RGB
            lut_a2b0.grid_points = [lut.shape[0]] * 3  # LUT size
            
            # Convert LUT to proper format
            lut_data = lut.astype(np.float32)
            lut_a2b0.clut = lut_data
            
            # Create input and output curves (linear)
            lut_a2b0.input_curves = [colour.ICC_Profile_Tag_Curve() for _ in range(3)]
            lut_a2b0.output_curves = [colour.ICC_Profile_Tag_Curve() for _ in range(3)]
            
            # Set linear curves
            for curve in lut_a2b0.input_curves + lut_a2b0.output_curves:
                curve.curve_type = 0  # Linear
                curve.curve_data = np.array([1.0, 0.0])  # y = x
            
            a2b0.data = lut_a2b0
            profile.tags['A2B0'] = a2b0
            
            # Create B2A0 tag (PCS to RGB transformation) - identity
            b2a0 = colour.ICC_Profile_Tag()
            b2a0.signature = b'B2A0'
            
            lut_b2a0 = colour.ICC_Profile_Tag_LUT_B2A0()
            lut_b2a0.input_channels = 3
            lut_b2a0.output_channels = 3
            lut_b2a0.grid_points = [lut.shape[0]] * 3
            
            # Identity LUT for B2A
            identity_lut = np.zeros_like(lut)
            for i in range(lut.shape[0]):
                for j in range(lut.shape[1]):
                    for k in range(lut.shape[2]):
                        identity_lut[i, j, k] = [i/(lut.shape[0]-1), j/(lut.shape[1]-1), k/(lut.shape[2]-1)]
            
            lut_b2a0.clut = identity_lut.astype(np.float32)
            lut_b2a0.input_curves = [colour.ICC_Profile_Tag_Curve() for _ in range(3)]
            lut_b2a0.output_curves = [colour.ICC_Profile_Tag_Curve() for _ in range(3)]
            
            for curve in lut_b2a0.input_curves + lut_b2a0.output_curves:
                curve.curve_type = 0
                curve.curve_data = np.array([1.0, 0.0])
            
            b2a0.data = lut_b2a0
            profile.tags['B2A0'] = b2a0
            
            # Write ICC profile to file
            with open(filename, 'wb') as f:
                profile.write(f)
                
        except Exception as e:
            logger.error(f"Error creating ICC profile: {e}")
            # Fallback: create a simple matrix-based ICC profile
            self.create_simple_matrix_icc_profile(filename, degree)
    
    def create_simple_matrix_icc_profile(self, filename, degree):
        """Create a simple matrix-based ICC profile as fallback."""
        try:
            # Create a basic ICC profile with just matrix transformation
            profile = colour.ICC_Profile()
            profile.profile_description = f"ColorChecker_Matrix_Degree_{degree}"
            profile.copyright = "Generated by ColorChecker Tool"
            
            # Create simple matrix transformation
            # This is a simplified version - in practice you'd want the actual correction matrix
            matrix = np.eye(3)  # Identity matrix as placeholder
            
            # Create A2B0 tag with matrix
            a2b0 = colour.ICC_Profile_Tag()
            a2b0.signature = b'A2B0'
            
            matrix_a2b0 = colour.ICC_Profile_Tag_Matrix_A2B0()
            matrix_a2b0.input_channels = 3
            matrix_a2b0.output_channels = 3
            matrix_a2b0.matrix = matrix.astype(np.float32)
            matrix_a2b0.offset = np.zeros(3, dtype=np.float32)
            
            # Create input curves (linear)
            matrix_a2b0.input_curves = [colour.ICC_Profile_Tag_Curve() for _ in range(3)]
            for curve in matrix_a2b0.input_curves:
                curve.curve_type = 0
                curve.curve_data = np.array([1.0, 0.0])
            
            a2b0.data = matrix_a2b0
            profile.tags['A2B0'] = a2b0
            
            # Write to file
            with open(filename, 'wb') as f:
                profile.write(f)
                
        except Exception as e:
            logger.error(f"Error creating simple ICC profile: {e}")
            raise ValueError("Failed to create ICC profile")

    def export_cube_lut(self):
        """Export a 3D LUT (.cube) file based on the current color correction."""
        method = self.correction_method_var.get()
        
        # Debug logging
        logger.info(f"Exporting 3D LUT with method: {method}")
        logger.info(f"current_polynomial_models: {self.current_polynomial_models}")
        logger.info(f"current_polynomial_degree: {self.current_polynomial_degree}")
        logger.info(f"current_matrix_transformation: {self.current_matrix_transformation}")
        
        if method == "Polynomial":
            if self.current_polynomial_models is None or self.current_polynomial_degree is None:
                messagebox.showwarning("Warning", "Please apply polynomial correction first to generate a 3D LUT.")
                return
        elif method == "Matrix":
            if self.current_matrix_transformation is None:
                messagebox.showwarning("Warning", "Please apply matrix correction first to generate a 3D LUT.")
                return
        else:
            messagebox.showwarning("Warning", "3D LUT export is currently only supported for polynomial and matrix correction.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".cube",
            filetypes=[("3D LUT files", "*.cube"), ("All files", "*.*")],
            title="Save 3D LUT (.cube)"
        )
        if not filename:
            return
        try:
            lut_size = 33  # Standard size
            
            if method == "Polynomial":
                logger.info(f"Creating polynomial LUT with models: {self.current_polynomial_models}, degree: {self.current_polynomial_degree}")
                lut = self.create_3d_lut_from_polynomial(self.current_polynomial_models, self.current_polynomial_degree, lut_size)
                logger.info(f"Polynomial LUT created: {lut is not None}")
            elif method == "Matrix":
                logger.info(f"Creating matrix LUT with transformation: {self.current_matrix_transformation}")
                lut = self.create_3d_lut_from_matrix(self.current_matrix_transformation, lut_size)
                logger.info(f"Matrix LUT created: {lut is not None}")
            else:
                messagebox.showerror("Error", f"Unsupported correction method: {method}")
                return
                
            if lut is None:
                messagebox.showerror("Error", "Failed to create 3D LUT - no valid correction data available.")
                return
                
            logger.info(f"LUT shape: {lut.shape if lut is not None else 'None'}")
            self.write_cube_file(filename, lut)
            messagebox.showinfo("Success", f"3D LUT (.cube) saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create 3D LUT:\n{str(e)}")
            logger.error(f"3D LUT export error: {e}")

    def write_cube_file(self, filename, lut):
        """Write a 3D LUT to a .cube file."""
        lut_size = lut.shape[0]
        with open(filename, 'w') as f:
            f.write(f"# Created by ColorChecker Tool\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n")
            f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n")
            # LUT data: order is blue fastest, then green, then red (OpenColorIO/Adobe order)
            for r in range(lut_size):
                for g in range(lut_size):
                    for b in range(lut_size):
                        rgb = lut[r, g, b]
                        f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

    def show_export_menu(self):
        """Show a popup menu for LUT export options."""
        # Create popup menu
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Export as .cube", command=self.export_cube_lut)
        menu.add_command(label="Export as .lut", command=self.export_lut_file)
        
        # Get button position and show menu
        x = self.export_lut_button.winfo_rootx()
        y = self.export_lut_button.winfo_rooty() + self.export_lut_button.winfo_height()
        menu.post(x, y)
    
    def export_lut_file(self):
        """Export a 3D LUT (.lut) file based on the current color correction."""
        method = self.correction_method_var.get()
        
        if method == "Polynomial":
            if self.current_polynomial_models is None or self.current_polynomial_degree is None:
                messagebox.showwarning("Warning", "Please apply polynomial correction first to generate a 3D LUT.")
                return
        elif method == "Matrix":
            if self.current_matrix_transformation is None:
                messagebox.showwarning("Warning", "Please apply matrix correction first to generate a 3D LUT.")
                return
        else:
            messagebox.showwarning("Warning", "3D LUT export is currently only supported for polynomial and matrix correction.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".lut",
            filetypes=[("LUT files", "*.lut"), ("All files", "*.*")],
            title="Export 3D LUT (.lut)"
        )
        if filename:
            try:
                # Generate 3D LUT based on correction method
                if method == "Polynomial":
                    lut = self.create_3d_lut_from_polynomial(
                        self.current_polynomial_models, 
                        self.current_polynomial_degree
                    )
                elif method == "Matrix":
                    lut = self.create_3d_lut_from_matrix(
                        self.current_matrix_transformation
                    )
                else:
                    messagebox.showerror("Error", f"Unsupported correction method: {method}")
                    return
                
                if lut is None:
                    messagebox.showerror("Error", "Failed to create 3D LUT - no valid correction data available.")
                    return
                
                # Write .lut file
                self.write_lut_file(filename, lut)
                
                messagebox.showinfo("Success", f"3D LUT exported successfully to:\n{filename}")
                logger.info(f"3D LUT exported to: {filename}")
                
            except Exception as e:
                error_msg = f"Error exporting 3D LUT: {str(e)}"
                messagebox.showerror("Error", error_msg)
                logger.error(error_msg)

    def write_lut_file(self, filename, lut):
        """Write a 3D LUT to a .lut file."""
        lut_size = lut.shape[0]
        
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# 3D LUT generated by ColorChecker Tool\n")
            f.write(f"# Size: {lut_size}x{lut_size}x{lut_size}\n")
            f.write(f"# Format: RGB to RGB\n")
            f.write(f"# Domain: 0.0 to 1.0\n\n")
            
            # Write LUT data
            for r in range(lut_size):
                for g in range(lut_size):
                    for b in range(lut_size):
                        # Get corrected RGB values
                        corrected_rgb = lut[r, g, b]
                        f.write(f"{corrected_rgb[0]:.6f} {corrected_rgb[1]:.6f} {corrected_rgb[2]:.6f}\n")

    def import_lut(self):
        """Import a 3D LUT file and apply it to the current image."""
        if self.original_image_rgb is None:
            messagebox.showwarning("Warning", "Please load an image first before importing a LUT.")
            return
            
        filename = filedialog.askopenfilename(
            title="Import 3D LUT",
            filetypes=[
                ("3D LUT files", "*.cube"),
                ("3D LUT files", "*.lut"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return
            
        try:
            # Load the LUT based on file extension
            if filename.lower().endswith('.cube'):
                lut = self.load_cube_lut(filename)
            elif filename.lower().endswith('.lut'):
                lut = self.load_lut_file(filename)
            else:
                messagebox.showerror("Error", "Unsupported file format. Please select a .cube or .lut file.")
                return
                
            if lut is None:
                messagebox.showerror("Error", "Failed to load LUT file.")
                return
                
            # Apply the LUT to the current image
            self.apply_lut_to_image(lut, filename)
            
        except Exception as e:
            error_msg = f"Error importing LUT: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def load_cube_lut(self, filename):
        """Load a 3D LUT from a .cube file."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            lut_size = None
            domain_min = [0.0, 0.0, 0.0]
            domain_max = [1.0, 1.0, 1.0]
            
            # Parse header
            data_start = 0
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif line.startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[1])
                elif line.startswith('DOMAIN_MIN'):
                    parts = line.split()[1:]
                    domain_min = [float(x) for x in parts]
                elif line.startswith('DOMAIN_MAX'):
                    parts = line.split()[1:]
                    domain_max = [float(x) for x in parts]
                else:
                    # First non-header line, start of data
                    data_start = i
                    break
            
            if lut_size is None:
                raise ValueError("LUT_3D_SIZE not found in .cube file")
            
            # Read LUT data
            lut_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        rgb = [float(x) for x in parts[:3]]
                        lut_data.append(rgb)
            
            if len(lut_data) != lut_size ** 3:
                raise ValueError(f"Expected {lut_size ** 3} entries, got {len(lut_data)}")
            
            # Reshape to 3D LUT (R, G, B, 3)
            lut = np.array(lut_data).reshape(lut_size, lut_size, lut_size, 3)
            
            logger.info(f"Loaded .cube LUT: {lut.shape}, domain: {domain_min} to {domain_max}")
            return lut
            
        except Exception as e:
            logger.error(f"Error loading .cube file: {e}")
            raise

    def load_lut_file(self, filename):
        """Load a 3D LUT from a .lut file."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Parse header to get LUT size
            lut_size = None
            data_start = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#'):
                    # Look for size information in comments
                    if 'size:' in line.lower() or 'size' in line.lower():
                        # Try to extract size from comment like "# Size: 33x33x33"
                        import re
                        size_match = re.search(r'(\d+)x(\d+)x(\d+)', line)
                        if size_match:
                            size = int(size_match.group(1))
                            if size == int(size_match.group(2)) == int(size_match.group(3)):
                                lut_size = size
                elif not line:
                    continue
                else:
                    # First non-comment, non-empty line, start of data
                    data_start = i
                    break
            
            # If we couldn't determine size from header, try to infer from data
            if lut_size is None:
                data_lines = [line for line in lines[data_start:] if line.strip() and not line.strip().startswith('#')]
                total_entries = len(data_lines)
                # Try common LUT sizes
                for size in [17, 33, 65, 129]:
                    if total_entries == size ** 3:
                        lut_size = size
                        break
                
                if lut_size is None:
                    raise ValueError(f"Could not determine LUT size from {total_entries} entries")
            
            # Read LUT data
            lut_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        rgb = [float(x) for x in parts[:3]]
                        lut_data.append(rgb)
            
            if len(lut_data) != lut_size ** 3:
                raise ValueError(f"Expected {lut_size ** 3} entries, got {len(lut_data)}")
            
            # Reshape to 3D LUT (R, G, B, 3)
            lut = np.array(lut_data).reshape(lut_size, lut_size, lut_size, 3)
            
            logger.info(f"Loaded .lut file: {lut.shape}")
            return lut
            
        except Exception as e:
            logger.error(f"Error loading .lut file: {e}")
            raise

    def apply_lut_to_image(self, lut, lut_filename):
        """Apply a 3D LUT to the current image with GPU acceleration and tiled processing."""
        try:
            if self.original_image_rgb is None:
                raise ValueError("No image loaded")
            
            start_time = time.time()
            
            # Get the original image
            image_rgb = self.original_image_rgb.astype(np.float32) / 255.0
            h, w, _ = image_rgb.shape
            
            # Show progress message
            lut_name = os.path.basename(lut_filename)
            self.root.title(f"ColorChecker - Applying LUT '{lut_name}'...")
            self.root.update()
            
            # Determine processing method based on user preference and image size
            image_size_mb = (h * w * 3 * 4) / (1024 * 1024)  # Size in MB
            corrected_rgb = None
            processing_method = self.lut_processing_method.get()
            
            # Apply user's processing method preference
            if processing_method == "GPU" and CUPY_AVAILABLE:
                logger.info("Using GPU acceleration as requested...")
                if image_size_mb > 100:
                    corrected_rgb = gpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024)
                else:
                    corrected_rgb = gpu_lut_interpolation(image_rgb, lut)
            
            elif processing_method == "CPU_Tiled":
                logger.info("Using CPU tiled processing as requested...")
                num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers
                corrected_rgb = cpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024, num_workers=num_workers)
            
            elif processing_method == "CPU_Standard":
                logger.info("Using standard CPU processing as requested...")
                if self.optimization_level.get() == "Fast":
                    corrected_rgb = self._optimize_lut_interpolation(image_rgb, lut)
                else:
                    try:
                        lut_size = lut.shape[0]
                        lut3d = colour.LUT3D(table=lut, size=lut_size, name=lut_name)
                        corrected_rgb = lut3d.apply(image_rgb)
                    except AttributeError:
                        corrected_rgb = self._optimize_lut_interpolation(image_rgb, lut)
            
            else:  # Auto mode - intelligent selection
                logger.info(f"Auto mode: Large image detected ({image_size_mb:.1f}MB)")
                
                # For very large images (>100MB), use tiled processing
                if image_size_mb > 100:
                    # Try GPU tiled processing first
                    if CUPY_AVAILABLE:
                        logger.info("Attempting GPU tiled processing...")
                        corrected_rgb = gpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024)
                    
                    # Fallback to CPU tiled processing
                    if corrected_rgb is None:
                        logger.info("Using CPU tiled processing with multiprocessing...")
                        num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers
                        corrected_rgb = cpu_lut_interpolation_tiled(image_rgb, lut, tile_size=1024, num_workers=num_workers)
                
                # For medium images, try GPU acceleration
                elif image_size_mb > 10 and CUPY_AVAILABLE:
                    logger.info("Attempting GPU acceleration...")
                    corrected_rgb = gpu_lut_interpolation(image_rgb, lut)
                
                # Fallback to optimized CPU processing
                if corrected_rgb is None:
                    logger.info("Using optimized CPU processing...")
                    if self.optimization_level.get() == "Fast":
                        corrected_rgb = self._optimize_lut_interpolation(image_rgb, lut)
                    else:
                        try:
                            lut_size = lut.shape[0]
                            lut3d = colour.LUT3D(table=lut, size=lut_size, name=lut_name)
                            corrected_rgb = lut3d.apply(image_rgb)
                        except AttributeError:
                            corrected_rgb = self._optimize_lut_interpolation(image_rgb, lut)
            
            # Convert back to uint8
            corrected_rgb = np.clip(corrected_rgb * 255, 0, 255).astype(np.uint8)
            
            # Store the corrected image and LUT for before/after functionality
            self.corrected_image = corrected_rgb
            self.current_lut = lut
            self.current_lut_filename = lut_filename
            
            # Update display
            self.display_corrected_image()
            
            # Reset window title
            self.root.title("ColorChecker - Color Correction Tool")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Applied LUT '{lut_name}' to image: {image_rgb.shape} in {elapsed_time:.2f} seconds")
            
            # Show success message with timing and method used
            if processing_method == "GPU":
                method_info = "GPU tiled" if image_size_mb > 100 else "GPU accelerated"
            elif processing_method == "CPU_Tiled":
                method_info = "CPU tiled processing"
            elif processing_method == "CPU_Standard":
                method_info = "CPU standard processing"
            else:  # Auto mode
                method_info = "GPU tiled" if image_size_mb > 100 and CUPY_AVAILABLE else \
                             "GPU accelerated" if image_size_mb > 10 and CUPY_AVAILABLE else \
                             "CPU tiled" if image_size_mb > 100 else \
                             "CPU optimized"
            
            messagebox.showinfo("Success", 
                              f"3D LUT '{lut_name}' applied successfully!\n"
                              f"Processing time: {elapsed_time:.2f} seconds\n"
                              f"Method: {method_info}\n"
                              f"Image size: {image_size_mb:.1f}MB")
            
        except Exception as e:
            error_msg = f"Error applying LUT: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
            # Reset window title on error
            self.root.title("ColorChecker - Color Correction Tool")

    def apply_lut_interpolation(self, image_rgb, lut):
        """Apply 3D LUT to image using vectorized trilinear interpolation."""
        lut_size = lut.shape[0]
        
        # Create coordinate grids
        r_coords = image_rgb[:, :, 0] * (lut_size - 1)
        g_coords = image_rgb[:, :, 1] * (lut_size - 1)
        b_coords = image_rgb[:, :, 2] * (lut_size - 1)
        
        # Get integer and fractional parts
        r0 = np.floor(r_coords).astype(int)
        g0 = np.floor(g_coords).astype(int)
        b0 = np.floor(b_coords).astype(int)
        
        r1 = np.minimum(r0 + 1, lut_size - 1)
        g1 = np.minimum(g0 + 1, lut_size - 1)
        b1 = np.minimum(b0 + 1, lut_size - 1)
        
        # Fractional parts for interpolation
        r_frac = r_coords - r0
        g_frac = g_coords - g0
        b_frac = b_coords - b0
        
        # Get the 8 corners of the cube using vectorized indexing
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]
        
        # Interpolation weights (vectorized)
        w000 = (1 - r_frac) * (1 - g_frac) * (1 - b_frac)
        w001 = (1 - r_frac) * (1 - g_frac) * b_frac
        w010 = (1 - r_frac) * g_frac * (1 - b_frac)
        w011 = (1 - r_frac) * g_frac * b_frac
        w100 = r_frac * (1 - g_frac) * (1 - b_frac)
        w101 = r_frac * (1 - g_frac) * b_frac
        w110 = r_frac * g_frac * (1 - b_frac)
        w111 = r_frac * g_frac * b_frac
        
        # Interpolate (vectorized)
        corrected_rgb = (w000[..., np.newaxis] * c000 + 
                        w001[..., np.newaxis] * c001 + 
                        w010[..., np.newaxis] * c010 + 
                        w011[..., np.newaxis] * c011 +
                        w100[..., np.newaxis] * c100 + 
                        w101[..., np.newaxis] * c101 + 
                        w110[..., np.newaxis] * c110 + 
                        w111[..., np.newaxis] * c111)
        
        return corrected_rgb

    # ============================================================================
    # PERFORMANCE OPTIMIZATION METHODS
    # ============================================================================
    
    def _get_correction_cache_key(self, measured_labs, reference_labs, method):
        """Generate a cache key for correction results."""
        import hashlib
        # Create a hash of the input data
        data_str = str(measured_labs) + str(reference_labs) + method
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _apply_cached_correction(self, image_rgb, cached_result):
        """Apply cached correction to image."""
        method = cached_result['method']
        
        if method == 'polynomial':
            return self._apply_polynomial_correction_optimized(
                image_rgb, 
                cached_result['models'], 
                cached_result['degree'], 
                None  # measured_labs_normalized not needed for cached
            )
        elif method == 'matrix':
            matrix = cached_result['matrix']
            image_rgb_norm = image_rgb.astype(np.float32) / 255.0
            h, w, _ = image_rgb_norm.shape
            image_reshaped = image_rgb_norm.reshape(-1, 3)
            corrected_reshaped = image_reshaped @ matrix
            corrected_rgb = corrected_reshaped.reshape(h, w, 3)
            corrected_rgb = self._apply_brightness_preservation(image_rgb_norm, corrected_rgb)
            return np.clip(corrected_rgb * 255, 0, 255).astype(np.uint8)
        
        return image_rgb
    
    def _normalize_lab_vectorized(self, lab_array):
        """Vectorized LAB normalization."""
        normalized = lab_array.copy().astype(np.float32)
        # Normalize L* to [0, 1]
        normalized[:, 0] = normalized[:, 0] / 100.0
        # Normalize a* and b* to [0, 1]
        normalized[:, 1:] = (normalized[:, 1:] + 128.0) / 255.0
        return normalized
    
    def _denormalize_lab_vectorized(self, lab_normalized):
        """Vectorized LAB denormalization."""
        denormalized = lab_normalized.copy().astype(np.float32)
        # Denormalize L* from [0, 1] to [0, 100]
        denormalized[:, 0] = denormalized[:, 0] * 100.0
        # Denormalize a* and b* from [0, 1] to [-128, 127]
        denormalized[:, 1:] = denormalized[:, 1:] * 255.0 - 128.0
        return denormalized
    
    def _apply_polynomial_correction_optimized(self, image_rgb, models, degree, measured_labs_normalized):
        """Optimized polynomial correction application."""
        # Convert to LAB and normalize
        image_lab = rgb2lab(image_rgb.astype(np.float32) / 255.0)
        h, w, _ = image_lab.shape
        
        # Vectorized normalization
        image_lab_normalized = self._normalize_lab_vectorized(image_lab.reshape(-1, 3))
        
        # Apply correction based on degree
        if degree == 1:
            # Linear correction - fastest
            predicted_transformation = np.zeros_like(image_lab_normalized)
            for i in range(3):
                predicted_transformation[:, i] = models[i].predict(image_lab_normalized)
        elif degree > 1:
            # Polynomial correction with caching
            if hasattr(self, 'current_polynomial_transformer') and self.current_polynomial_transformer is not None:
                features = self.current_polynomial_transformer.transform(image_lab_normalized)
            else:
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=degree, include_bias=True)
                features = poly.fit_transform(image_lab_normalized)
            
            predicted_transformation = np.zeros_like(image_lab_normalized)
            for i in range(3):
                predicted_transformation[:, i] = models[i].predict(features)
        elif degree == -1:
            # Advanced features
            print("Applying advanced features correction...")
            advanced_features = self.create_advanced_features_for_prediction(image_lab_normalized)
            predicted_transformation = np.zeros_like(image_lab_normalized)
            for i in range(3):
                predicted_transformation[:, i] = models[i].predict(advanced_features)
        elif degree == -2:
            # Multi-scale correction
            print("Applying multi-scale correction...")
            predicted_transformation = self.apply_multi_scale_correction(image_lab_normalized, models, degree, measured_labs_normalized)
        else:
            # Identity transformation
            predicted_transformation = np.zeros_like(image_lab_normalized)
        
        # Debug: Check predicted transformation values
        print(f"Predicted transformation stats: min={np.min(predicted_transformation):.6f}, max={np.max(predicted_transformation):.6f}, mean={np.mean(predicted_transformation):.6f}")
        print(f"Sample predicted transformations: {predicted_transformation[:3]}")
        
        # Add transformation and denormalize
        corrected_flat = image_lab_normalized + predicted_transformation
        
        # Debug: Check transformation values
        if np.any(np.abs(predicted_transformation) > 0.1):
            print(f"Large transformation detected: max={np.max(np.abs(predicted_transformation)):.4f}")
        
        # Clip the corrected normalized values to valid ranges before denormalizing
        corrected_flat[:, 0] = np.clip(corrected_flat[:, 0], 0, 1)  # L* normalized
        corrected_flat[:, 1:] = np.clip(corrected_flat[:, 1:], 0, 1)  # a*, b* normalized
        
        corrected_lab = self._denormalize_lab_vectorized(corrected_flat)
        
        # Clip to valid LAB ranges
        corrected_lab[:, 0] = np.clip(corrected_lab[:, 0], 0, 100)
        corrected_lab[:, 1:] = np.clip(corrected_lab[:, 1:], -128, 127)
        
        # Reshape and convert back to RGB
        corrected_lab = corrected_lab.reshape(h, w, 3)
        corrected_rgb = lab2rgb(corrected_lab)
        corrected_rgb = np.clip(corrected_rgb, 0, 1)
        
        return (corrected_rgb * 255).astype(np.uint8)
    
    def _apply_brightness_preservation(self, original_rgb, corrected_rgb):
        """Apply brightness preservation with vectorized operations."""
        # Clip to valid RGB range
        corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
        
        # Calculate brightness preservation
        original_brightness = np.mean(original_rgb)
        corrected_brightness = np.mean(corrected_rgb)
        
        # If brightness dropped too much, apply gentle boost
        if corrected_brightness < original_brightness * 0.85:  # 15% threshold
            brightness_ratio = original_brightness / corrected_brightness
            boost_factor = min(brightness_ratio * 0.8, 1.5)  # Cap at 1.5x
            corrected_rgb = np.clip(corrected_rgb * boost_factor, 0.0, 1.0)
        
        return corrected_rgb
    
    def _optimize_lut_interpolation(self, image_rgb, lut):
        """Optimized LUT interpolation using vectorized operations."""
        lut_size = lut.shape[0]
        
        # Create coordinate grids (vectorized)
        r_coords = image_rgb[:, :, 0] * (lut_size - 1)
        g_coords = image_rgb[:, :, 1] * (lut_size - 1)
        b_coords = image_rgb[:, :, 2] * (lut_size - 1)
        
        # Get integer and fractional parts (vectorized)
        r0 = np.floor(r_coords).astype(int)
        g0 = np.floor(g_coords).astype(int)
        b0 = np.floor(b_coords).astype(int)
        
        r1 = np.minimum(r0 + 1, lut_size - 1)
        g1 = np.minimum(g0 + 1, lut_size - 1)
        b1 = np.minimum(b0 + 1, lut_size - 1)
        
        # Fractional parts for interpolation
        r_frac = r_coords - r0
        g_frac = g_coords - g0
        b_frac = b_coords - b0
        
        # Get the 8 corners using advanced indexing
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]
        
        # Interpolation weights (vectorized)
        w000 = (1 - r_frac) * (1 - g_frac) * (1 - b_frac)
        w001 = (1 - r_frac) * (1 - g_frac) * b_frac
        w010 = (1 - r_frac) * g_frac * (1 - b_frac)
        w011 = (1 - r_frac) * g_frac * b_frac
        w100 = r_frac * (1 - g_frac) * (1 - b_frac)
        w101 = r_frac * (1 - g_frac) * b_frac
        w110 = r_frac * g_frac * (1 - b_frac)
        w111 = r_frac * g_frac * b_frac
        
        # Interpolate (vectorized)
        corrected_rgb = (w000[..., np.newaxis] * c000 + 
                        w001[..., np.newaxis] * c001 + 
                        w010[..., np.newaxis] * c010 + 
                        w011[..., np.newaxis] * c011 +
                        w100[..., np.newaxis] * c100 + 
                        w101[..., np.newaxis] * c101 + 
                        w110[..., np.newaxis] * c110 + 
                        w111[..., np.newaxis] * c111)
        
        return corrected_rgb
    
    def clear_correction_cache(self):
        """Clear the correction cache to free memory."""
        if hasattr(self, '_correction_cache'):
            self._correction_cache.clear()
            logger.info("Correction cache cleared")
    
    def get_performance_stats(self):
        """Get performance statistics."""
        stats = {
            'cache_size': len(self._correction_cache) if hasattr(self, '_correction_cache') else 0,
            'numba_available': NUMBA_AVAILABLE,
            'cupy_available': CUPY_AVAILABLE,
            'memory_usage': 'N/A'  # Could add memory monitoring here
        }
        return stats



def main():
    """
    Main entry point for the ColorChecker application.
    
    Initializes the Tkinter root window and starts the color correction application.
    """
    root = tk.Tk()
    app = ColorCorrectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 
