import sys
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path for importing main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from vegcover.models import AnalysisResult, Box, CategoryStat, CoverageStat, DetectResult, StitchResult, VegIndexMap


class TestGradioUI:
    def test_ui_import(self):
        """Should be able to import the main module."""
        import main
        assert hasattr(main, "create_ui")

    def test_create_ui_returns_blocks(self):
        """create_ui() should return a Gradio Blocks instance."""
        import gradio as gr
        from main import create_ui

        ui = create_ui()
        assert isinstance(ui, gr.Blocks)
