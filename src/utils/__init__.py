"""
Módulo de utilitários para processamento OMR (Optical Mark Recognition).
"""

from .aruco_detector import detect_sheet, order_ids
from .sheet_geometry import base_centers, group_sir, _scaled_mm
from .choice_detector import detect_choices, ratio_fill, choose
from .visualization import (
    draw_result, centers_preview, info_panel, 
    board, fit_h, wrap_lines
)
from .file_handler import (
    save_img, save_metadata, create_log_data, save_all_images
)

__all__ = [
    # ArUco detection
    'detect_sheet',
    'order_ids',
    
    # Sheet geometry
    'base_centers',
    'group_sir',
    
    # Choice detection
    'detect_choices',
    'ratio_fill',
    'choose',
    
    # Visualization
    'draw_result',
    'centers_preview',
    'info_panel',
    'board',
    'fit_h',
    'wrap_lines',
    
    # File handling
    'save_img',
    'save_metadata',
    'create_log_data',
    'save_all_images',
]
