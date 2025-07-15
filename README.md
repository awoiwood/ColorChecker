# ColorFix - Advanced Color Correction Tool

A comprehensive color correction application that uses ColorChecker charts to calibrate and correct color accuracy in images. Supports multiple correction methods including polynomial regression, matrix transformation, and 3D LUT.

## Features

- **Multiple Correction Algorithms**: Polynomial regression, Matrix transformation, and 3D LUT
- **Advanced Optimization**: Robust regression and cross-validation
- **Multi-scale Processing**: Optimized for different image scales and color spaces
- **Skin Tone Priority**: Specialized optimization for skin tones
- **Greyscale Optimization**: Enhanced accuracy for neutral colors
- **Real-time Color Difference Calculation**: CIEDE2000, CIEDE94, CIEDE76 metrics
- **Interactive GUI**: Zoom, pan, and before/after comparison
- **Batch Processing**: Process multiple images efficiently
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Export Options**: ICC profiles, CUBE LUT files, and detailed reports

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- NumPy
- SciPy
- scikit-learn
- PIL (Pillow)
- tkinter (usually included with Python)
- colour-science
- PyOpenColorIO
- reportlab

### Optional Dependencies

- **Numba**: For JIT compilation performance improvements
- **CuPy**: For GPU acceleration (requires CUDA-compatible GPU)

### Install Dependencies

```bash
pip install opencv-python numpy scipy scikit-learn pillow colour-science PyOpenColorIO reportlab
```

For optional GPU acceleration:
```bash
pip install numba cupy-cuda11x  # Replace with appropriate CUDA version
```

## Usage

### Running the Application

```bash
python "color checker.py"
```

### Basic Workflow

1. **Load Image**: Open an image containing a ColorChecker chart
2. **Select Reference Colors**: Click on the ColorChecker patches to measure their colors
3. **Choose Correction Method**: Select from Polynomial, Matrix, or LUT correction
4. **Apply Correction**: Process the image with the selected method
5. **Export Results**: Save corrected images, ICC profiles, or LUT files

### Correction Methods

#### Polynomial Regression
- Uses polynomial features for non-linear color transformations
- Supports multiple degrees (1-5) for different complexity levels
- Includes robust regression options (Huber, RANSAC)

#### Matrix Transformation
- Linear color space transformation
- Fast and efficient for simple corrections
- Good for basic color calibration

#### 3D Look-Up Table (LUT)
- High-precision color mapping
- Supports various LUT formats (CUBE, custom)
- GPU-accelerated interpolation available

## Building with PyInstaller

To create a standalone executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed "color checker.py"
```

## Performance Optimization

The application includes several optimization features:

- **GPU Acceleration**: Automatic detection and use of CUDA-compatible GPUs
- **Multi-threading**: Parallel processing for large images
- **Memory Management**: Efficient handling of large image files
- **Caching**: Intelligent caching of correction results

## Export Options

### ICC Profiles
Generate ICC profiles for use in other applications:
- Matrix-based profiles for simple corrections
- LUT-based profiles for complex transformations

### CUBE LUT Files
Export 3D LUT files compatible with:
- DaVinci Resolve
- Adobe Premiere Pro
- Final Cut Pro
- Other professional video editing software

### PDF Reports
Generate detailed correction reports including:
- Before/after color comparisons
- Delta E measurements
- Correction parameters
- Performance statistics

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.

## Acknowledgments

- ColorChecker charts by X-Rite
- OpenCV for computer vision capabilities
- scikit-learn for machine learning algorithms
- colour-science for color space transformations 