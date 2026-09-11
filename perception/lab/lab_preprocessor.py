"""
lab_preprocessor.py
===================
Stage 1: CPU-Bound Tabular Document Preprocessing & Adaptive Chunking.
Strictly pinned to Host CPU (OpenCV / Pillow / NumPy) - Zero VRAM Footprint.

Mitigations for 8GB VRAM & Small Vision-Language Models (Qwen2.5-VL-3B):
1. Avoids A4 300-DPI Vision Patch Token Explosion:
   An uncompressed 2480x3508 A4 scan consumes ~4,000 vision tokens, saturating
   KV cache and triggering CUDA OOM on an RTX 4060 (8GB).
2. Preserves Column-to-Header Semantic Association:
   Slices tabular data into overlapping horizontal strips (5-8 rows each) and
   dynamically stamps the extracted table header onto every subsequent strip,
   preventing the VLM from hallucinating or swapping column assignments.
"""

import os
import io
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import cv2
from PIL import Image

try:
    import pymupdf as fitz  # PyMuPDF for high-fidelity, memory-efficient PDF rendering
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False


@dataclass
class LabImageChunk:
    """Represents a preprocessed horizontal table strip for VLM ingestion."""
    chunk_index: int
    total_chunks: int
    image_bgr: np.ndarray
    row_start_idx: int
    row_end_idx: int
    has_prepended_header: bool
    original_bbox: Tuple[int, int, int, int]  # (ymin, ymax, xmin, xmax)
    token_estimated_count: int


class LabPreprocessor:
    """
    CPU-bound document preprocessor for tabular clinical laboratory reports.
    Executes deskewing, horizontal projection profiling, header extraction,
    and adaptive horizontal strip slicing.
    """

    def __init__(
        self,
        rows_per_chunk: int = 25,
        overlap_rows: int = 2,
        target_strip_width: int = 1120,
        max_chunk_height: int = 2200,
        render_pdf_dpi: int = 175
    ):
        self.rows_per_chunk = rows_per_chunk
        self.overlap_rows = overlap_rows
        self.target_strip_width = target_strip_width
        self.max_chunk_height = max_chunk_height
        self.render_pdf_dpi = render_pdf_dpi

    # =========================================================================
    # 1. Image Loading & PDF Normalization
    # =========================================================================
    def load_document(self, doc_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads document input into a BGR NumPy array.
        Handles PDF multi-page or single page rendering at optimal DPI (175 DPI)
        to balance font legibility and memory footprint.
        """
        if isinstance(doc_input, np.ndarray):
            if len(doc_input.shape) == 2:
                return cv2.cvtColor(doc_input, cv2.COLOR_GRAY2BGR)
            return doc_input

        if isinstance(doc_input, Image.Image):
            rgb = np.array(doc_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Byte stream or file path
        if isinstance(doc_input, bytes):
            # Check if PDF magic header '%PDF'
            if doc_input.startswith(b"%PDF"):
                return self._render_pdf_bytes_to_bgr(doc_input)
            np_arr = np.frombuffer(doc_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes.")
            return img

        if isinstance(doc_input, str):
            if not os.path.exists(doc_input):
                raise FileNotFoundError(f"Document file not found: {doc_input}")

            if doc_input.lower().endswith(".pdf"):
                with open(doc_input, "rb") as f:
                    return self._render_pdf_bytes_to_bgr(f.read())

            img = cv2.imread(doc_input, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"OpenCV failed to read image path: {doc_input}")
            return img

        raise TypeError(f"Unsupported document input type: {type(doc_input)}")

    def _render_pdf_bytes_to_bgr(self, pdf_bytes: bytes) -> np.ndarray:
        """Renders first page (or vertically concats primary lab page) of PDF via PyMuPDF."""
        if not HAS_FITZ:
            raise RuntimeError("PyMuPDF (fitz) is required to render PDF lab reports. Install via: pip install pymupdf")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("Empty PDF document.")

        page = doc[0]  # Lab results typically start on Page 1
        zoom = self.render_pdf_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        # Convert RGB to BGR
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # =========================================================================
    # 2. Skew Detection & High-Fidelity Deskewing
    # =========================================================================
    def deskew_page(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects document skew angle using Hough line transform and text orientation.
        Rotates page with clean white padding to eliminate artificial black border noise.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Inverted Otsu threshold for text and line extraction
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect prominent lines via Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi / 180,
            threshold=120,
            minLineLength=w // 6,
            maxLineGap=20
        )

        angles = []
        if lines is not None:
            for line in lines:
                pts = line.ravel()
                if len(pts) >= 4:
                    x1, y1, x2, y2 = pts[:4]
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dx) > 0.001:
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        # Only consider near-horizontal text lines (-40 to +40 degrees)
                        if -45.0 <= angle_deg <= 45.0:
                            angles.append(angle_deg)

        if not angles:
            # Fallback to minimum area rectangle bounding all text pixels
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) > 50:
                rect = cv2.minAreaRect(coords)
                box_angle = rect[-1]
                if box_angle < -45:
                    box_angle = -(90 + box_angle)
                elif box_angle > 45:
                    box_angle = 90 - box_angle
                if -45.0 <= box_angle <= 45.0:
                    angles.append(box_angle)

        median_angle = float(np.median(angles)) if angles else 0.0

        # Avoid jitter for tiny sub-degree tilts
        if abs(median_angle) < 0.35:
            return image, 0.0

        # Rotate document with border value set to white (255, 255, 255)
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return deskewed, median_angle

    # =========================================================================
    # 3. Horizontal Projection Profile (HPP) & Table Row Gutter Detection
    # =========================================================================
    def compute_horizontal_projection_profile(self, binary_img: np.ndarray) -> np.ndarray:
        """
        Computes the 1D horizontal ink density profile.
        Values near zero correspond to white gutters (inter-row spaces).
        High values indicate text rows or horizontal table border lines.
        """
        # Ensure ink is 1 and background is 0
        ink_map = (binary_img > 0).astype(np.float32)
        hpp = np.sum(ink_map, axis=1)

        # Smooth with a 1D Gaussian kernel to avoid character-stroke spikes
        kernel_size = 7
        sigma = 2.0
        kernel = cv2.getGaussianKernel(kernel_size, sigma).ravel()
        smoothed_hpp = np.convolve(hpp, kernel, mode="same")
        return smoothed_hpp

    def detect_table_grid_and_gutters(self, image: np.ndarray) -> Tuple[List[int], Optional[Tuple[int, int]]]:
        """
        Identifies horizontal row boundaries and the table header bounding interval.
        Works seamlessly on both ruled tables (with grid lines) and borderless tables.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Adaptive thresholding to highlight text and lines
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )

        # Step A: Check for explicit horizontal grid lines using morphological kernel
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
        horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
        line_proj = np.sum(horiz_lines > 0, axis=1)
        explicit_line_indices = np.where(line_proj > (w * 0.25))[0]

        row_dividers = []
        if len(explicit_line_indices) >= 3:
            # Cluster line indices that are adjacent
            current_cluster = [explicit_line_indices[0]]
            for idx in explicit_line_indices[1:]:
                if idx - current_cluster[-1] <= 6:
                    current_cluster.append(idx)
                else:
                    row_dividers.append(int(np.mean(current_cluster)))
                    current_cluster = [idx]
            if current_cluster:
                row_dividers.append(int(np.mean(current_cluster)))

        # Step B: If ruled lines are sparse (borderless report), use Horizontal Projection Profile
        if len(row_dividers) < 5:
            hpp = self.compute_horizontal_projection_profile(thresh)
            # Normalize profile
            max_val = np.max(hpp) if np.max(hpp) > 0 else 1.0
            norm_hpp = hpp / max_val

            # Valleys where ink is minimal (< 5% of peak) represent row gutters
            gutter_threshold = 0.05
            is_gutter = norm_hpp < gutter_threshold

            gutter_transitions = []
            in_gutter = False
            start_g = 0
            for y, val in enumerate(is_gutter):
                if val and not in_gutter:
                    in_gutter = True
                    start_g = y
                elif not val and in_gutter:
                    in_gutter = False
                    gutter_transitions.append((start_g + y) // 2)

            # Filter gutters: rows must be separated by reasonable line height (14px to 100px)
            filtered_dividers = [0]
            for g_idx in gutter_transitions:
                if g_idx - filtered_dividers[-1] >= 16:
                    filtered_dividers.append(g_idx)
            if h - filtered_dividers[-1] > 20:
                filtered_dividers.append(h)
            row_dividers = filtered_dividers

        # Ensure bounds
        if not row_dividers or row_dividers[0] != 0:
            row_dividers.insert(0, 0)
        if row_dividers[-1] != h:
            row_dividers.append(h)

        # Header detection: Scan top 35% of page for candidate header row
        header_bounds = None
        if len(row_dividers) >= 3:
            # Header is typically between the first and second/third row dividers
            y1 = row_dividers[0]
            # Find the divider that captures the top header section (usually between 35px and 180px)
            for i in range(1, min(len(row_dividers), 5)):
                y2 = row_dividers[i]
                if 28 <= (y2 - y1) <= 220:
                    header_bounds = (y1, y2)
                    break

        return row_dividers, header_bounds

    # =========================================================================
    # 4. Adaptive Strip Slicing with Header Preservation
    # =========================================================================
    def slice_table_into_strips(self, image: np.ndarray) -> List[LabImageChunk]:
        """
        Partitions the lab report into contextual horizontal strips:
        - Maintains 5-8 rows per chunk.
        - Employs 1-2 row overlap between consecutive strips to prevent boundary cutoff.
        - Stitches the extracted table header strip at the top of every subsequent chunk.
        - Dynamically resizes strips to maintain crisp OCR legibility while capping vision tokens.
        """
        deskewed, angle = self.deskew_page(image)
        h, w = deskewed.shape[:2]

        row_dividers, header_bounds = self.detect_table_grid_and_gutters(deskewed)

        # Extract Header Strip image
        header_strip: Optional[np.ndarray] = None
        header_height = 0
        if header_bounds is not None:
            hy1, hy2 = header_bounds
            if hy2 > hy1 + 10:
                header_strip = deskewed[hy1:hy2, 0:w].copy()
                header_height = header_strip.shape[0]

        # Calculate data row intervals (excluding the standalone header)
        data_dividers = [d for d in row_dividers if header_bounds is None or d >= header_bounds[1]]
        if len(data_dividers) < 2:
            data_dividers = row_dividers

        num_intervals = len(data_dividers) - 1
        chunks: List[LabImageChunk] = []

        if h <= self.max_chunk_height or num_intervals <= self.rows_per_chunk:
            # Document is compact enough to process in a single high-context strip
            chunk_img = self._rescale_strip(deskewed, self.target_strip_width)
            token_est = self.estimate_vlm_tokens(chunk_img.shape[1], chunk_img.shape[0])
            chunks.append(LabImageChunk(
                chunk_index=0,
                total_chunks=1,
                image_bgr=chunk_img,
                row_start_idx=0,
                row_end_idx=max(1, num_intervals),
                has_prepended_header=False,
                original_bbox=(0, h, 0, w),
                token_estimated_count=token_est
            ))
            return chunks

        # Multi-chunk sliding window with overlap
        step = max(1, self.rows_per_chunk - self.overlap_rows)
        chunk_idx = 0
        start_row = 0

        while start_row < num_intervals:
            end_row = min(num_intervals, start_row + self.rows_per_chunk)
            ymin = data_dividers[start_row]
            ymax = data_dividers[end_row]

            # Crop data slice
            data_slice = deskewed[ymin:ymax, 0:w]

            # Prepend Header to chunks beyond the first chunk if header was found
            has_header_prepended = False
            if header_strip is not None and start_row > 0:
                # Add a thin distinct divider line between header and data rows
                divider_line = np.full((3, w, 3), 180, dtype=np.uint8)
                combined = cv2.vconcat([header_strip, divider_line, data_slice])
                has_header_prepended = True
            else:
                combined = data_slice

            # Rescale strip to target width for optimal vision patch count & font clarity
            processed_strip = self._rescale_strip(combined, self.target_strip_width)
            token_est = self.estimate_vlm_tokens(processed_strip.shape[1], processed_strip.shape[0])

            chunks.append(LabImageChunk(
                chunk_index=chunk_idx,
                total_chunks=0,  # Will update after loop
                image_bgr=processed_strip,
                row_start_idx=start_row,
                row_end_idx=end_row,
                has_prepended_header=has_header_prepended,
                original_bbox=(ymin, ymax, 0, w),
                token_estimated_count=token_est
            ))

            chunk_idx += 1
            if end_row >= num_intervals:
                break
            start_row += step

        # Update total chunks count
        for ch in chunks:
            ch.total_chunks = len(chunks)

        return chunks

    def _rescale_strip(self, img: np.ndarray, target_w: int) -> np.ndarray:
        """
        Maintains aspect ratio while scaling to target width.
        Uses INTER_LANCZOS4 / INTER_CUBIC to preserve numerical precision and decimal points.
        """
        h, w = img.shape[:2]
        if w == target_w:
            return img

        scale = target_w / float(w)
        new_h = int(round(h * scale))

        # Enforce max height ceiling to prevent individual strips from overflowing token limit
        if new_h > self.max_chunk_height:
            scale = self.max_chunk_height / float(h)
            target_w = int(round(w * scale))
            new_h = self.max_chunk_height

        interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(img, (target_w, new_h), interpolation=interp)

    @staticmethod
    def estimate_vlm_tokens(width: int, height: int, patch_size: int = 14) -> int:
        """
        Estimates visual token count for Qwen2.5-VL.
        Qwen2.5-VL compresses 2x2 vision patches into 1 token: (H/14 * W/14) / 4.
        """
        patches_h = math.ceil(height / patch_size)
        patches_w = math.ceil(width / patch_size)
        vision_tokens = math.ceil((patches_h * patches_w) / 4)
        return vision_tokens

    @staticmethod
    def chunk_to_jpeg_bytes(chunk: LabImageChunk, quality: int = 95) -> bytes:
        """Encodes an image strip into compressed JPEG bytes for VLM consumption."""
        success, encoded = cv2.imencode(".jpg", chunk.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("Failed to encode chunk image to JPEG.")
        return encoded.tobytes()
