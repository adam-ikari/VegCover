"""VegCover - Aerial photo stitching and vegetation analysis tool.

Launch with: python main.py
"""

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from vegcover.models import AnalysisResult
from vegcover.pipeline import AnalysisPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_images(
    image_files: list,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    gsd: float | None = None,
) -> tuple:
    """Process uploaded images through the analysis pipeline.

    Args:
        image_files: List of uploaded image file paths.
        confidence_threshold: YOLO confidence threshold.
        iou_threshold: YOLO IoU threshold.
        gsd: Ground sampling distance (m/pixel).

    Returns:
        Tuple of (annotated_image, heatmap, pie_chart, bar_chart, csv_path, json_path)
    """
    if not image_files:
        return [None] * 6 + ["Please upload at least one image."]

    # Load images
    images = []
    for file_path in image_files:
        img = Image.open(file_path).convert("RGB")
        images.append(np.array(img))

    # Run pipeline
    pipeline = AnalysisPipeline()
    result = pipeline.run(images, gsd=gsd)

    # Prepare outputs
    outputs = _prepare_outputs(result)
    return outputs


def _prepare_outputs(result: AnalysisResult) -> tuple:
    """Prepare Gradio-compatible outputs from analysis result."""
    from vegcover.exporter import ResultExporter

    # Export to get file paths
    exporter = ResultExporter(output_dir="output")
    paths = exporter.export(result)

    # Load exported images
    annotated = Image.open(paths["annotated_image"])
    heatmap = Image.open(paths["heatmap"])

    # Load charts if available
    pie_chart = None
    bar_chart = None
    if "charts" in paths:
        charts = Image.open(paths["charts"])
        # Split the combined chart image
        w, h = charts.size
        pie_chart = charts.crop((0, 0, w // 2, h))
        bar_chart = charts.crop((w // 2, 0, w, h))

    return (
        annotated,
        heatmap,
        pie_chart,
        bar_chart,
        paths.get("stats_csv"),
        paths.get("stats_json"),
        f"Analysis complete! Total vegetation coverage: {result.coverage.total_veg_ratio:.1%}",
    )


def create_ui() -> gr.Blocks:
    """Create the Gradio web interface."""
    with gr.Blocks(title="VegCover - Aerial Vegetation Analysis") as app:
        gr.Markdown("# VegCover - Aerial Vegetation Analysis")
        gr.Markdown("Upload aerial photos to stitch, detect vegetation, and analyze coverage.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Upload Photos")
                image_input = gr.File(
                    label="Aerial Photos",
                    file_count="multiple",
                    file_types=["image"],
                )

                # Parameters
                gr.Markdown("### Parameters")
                with gr.Accordion("Detection Settings", open=False):
                    confidence_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    iou_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                        label="IoU Threshold"
                    )

                with gr.Accordion("Coverage Settings", open=False):
                    gsd_input = gr.Number(
                        value=None, label="GSD (m/pixel)",
                        info="Leave empty for pixel ratios only"
                    )

                analyze_btn = gr.Button("Start Analysis", variant="primary")

            with gr.Column(scale=2):
                # Results section
                gr.Markdown("### Results")
                with gr.Tabs():
                    with gr.TabItem("Annotated Panorama"):
                        annotated_output = gr.Image(label="Vegetation Detection")
                    with gr.TabItem("Vegetation Heatmap"):
                        heatmap_output = gr.Image(label="Vegetation Distribution")
                    with gr.TabItem("Coverage Charts"):
                        pie_output = gr.Image(label="Coverage Pie Chart")
                        bar_output = gr.Image(label="Coverage Bar Chart")
                    with gr.TabItem("Export Data"):
                        csv_output = gr.File(label="CSV Report")
                        json_output = gr.File(label="JSON Report")

                status_output = gr.Textbox(label="Status", interactive=False)

        # Wire up the button
        analyze_btn.click(
            fn=process_images,
            inputs=[image_input, confidence_slider, iou_slider, gsd_input],
            outputs=[
                annotated_output,
                heatmap_output,
                pie_output,
                bar_output,
                csv_output,
                json_output,
                status_output,
            ],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()
