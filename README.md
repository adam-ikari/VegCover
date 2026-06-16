# VegCover - Aerial Vegetation Analysis Tool

A desktop application for stitching aerial photos and analyzing vegetation coverage.

## Features

- **Photo Stitching**: Automatically stitch multiple aerial photos into a panorama
- **Vegetation Detection**: YOLO-based detection of vegetation types
- **Vegetation Index**: ExG and VARI index computation
- **Coverage Statistics**: Pixel-level coverage analysis with area estimation
- **Export**: Annotated images, heatmaps, charts, CSV, and JSON reports

## Installation

### Requirements

- Python >= 3.13
- PyQt6
- OpenCV
- Ultralytics (YOLO)
- NumPy
- Matplotlib
- Pillow

### Install from source

```bash
pip install -e .
```

## Usage

### Run the application

```bash
python main.py
```

### Workflow

1. Click "Select Photos..." to choose aerial photos
2. Adjust parameters (Confidence, IoU, GSD)
3. Click "Start Analysis"
4. View results in tabs:
   - **Annotated Panorama**: Vegetation detection overlay
   - **Vegetation Heatmap**: Distribution heatmap
   - **Coverage Charts**: Pie and bar charts
   - **Export Data**: Export CSV, JSON, and images

## Development

### Run tests

```bash
pytest tests/ -v
```

### Project Structure

```
VegCover/
├── main.py                  # Entry point
├── src/vegcover/
│   ├── __init__.py
│   ├── models.py            # Data models
│   ├── stitcher.py            # Photo stitching
│   ├── detector.py            # YOLO vegetation detection
│   ├── vegindex.py            # Vegetation index calculation
│   ├── stats.py               # Coverage statistics
│   ├── exporter.py            # Export module
│   └── pipeline.py            # Pipeline controller
└── tests/                     # Test suite
```

## Building Executables

### Using PyInstaller

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name VegCover main.py
```

### GitHub Actions CI/CD

The project includes GitHub Actions workflows for building executables on Windows, macOS, and Ubuntu. Push a tag starting with `v` to trigger the build:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## License

MIT
