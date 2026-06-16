"""Pipeline controller that orchestrates all analysis steps."""

import logging

import numpy as np

from vegcover.detector import VegetationDetector
from vegcover.exporter import ResultExporter
from vegcover.models import AnalysisResult
from vegcover.stats import CoverageCalculator
from vegcover.stitcher import PhotoStitcher
from vegcover.vegindex import VegetationIndexCalculator

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full analysis pipeline."""

    def __init__(
        self,
        stitcher: PhotoStitcher | None = None,
        detector: VegetationDetector | None = None,
        veg_indexer: VegetationIndexCalculator | None = None,
        calculator: CoverageCalculator | None = None,
        exporter: ResultExporter | None = None,
    ):
        self.stitcher = stitcher or PhotoStitcher()
        self.detector = detector or VegetationDetector()
        self.veg_indexer = veg_indexer or VegetationIndexCalculator()
        self.calculator = calculator or CoverageCalculator()
        self.exporter = exporter or ResultExporter()

    def run(
        self,
        images: list[np.ndarray],
        gsd: float | None = None,
        export: bool = True,
    ) -> AnalysisResult:
        """Run the full analysis pipeline.

        Args:
            images: List of RGB images.
            gsd: Ground sampling distance in meters per pixel (optional).
            export: Whether to export results to files.

        Returns:
            AnalysisResult with all analysis data.
        """
        logger.info("Starting analysis pipeline...")

        # Step 1: Stitch photos
        logger.info("Stitching photos...")
        stitch_result = self.stitcher.stitch(images)
        panorama = stitch_result.panorama

        # Step 2: Detect vegetation
        logger.info("Detecting vegetation...")
        detect_result = self.detector.detect(panorama)

        # Step 3: Compute vegetation indices
        logger.info("Computing vegetation indices...")
        veg_index = self.veg_indexer.compute(panorama)

        # Step 4: Calculate coverage statistics
        logger.info("Calculating coverage statistics...")
        coverage = self.calculator.calculate(panorama, detect_result, veg_index, gsd=gsd)

        # Build result
        result = AnalysisResult(
            stitch=stitch_result,
            detect=detect_result,
            veg_index=veg_index,
            coverage=coverage,
        )

        # Step 5: Export results
        if export:
            logger.info("Exporting results...")
            self.exporter.export(result)

        logger.info("Analysis complete!")
        return result
