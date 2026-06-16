import sys
from pathlib import Path

import pytest

# Add project root to path for importing main.py
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPyQtUI:
    def test_ui_import(self):
        """Should be able to import the main module."""
        import main
        assert hasattr(main, "VegCoverApp")
        assert hasattr(main, "main")

    def test_app_window_title(self):
        """App window should have correct title."""
        from PyQt6.QtWidgets import QApplication
        from main import VegCoverApp

        # QApplication needs to be created before any QWidget
        app = QApplication.instance() or QApplication(sys.argv)
        window = VegCoverApp()
        assert window.windowTitle() == "VegCover - Aerial Vegetation Analysis"
