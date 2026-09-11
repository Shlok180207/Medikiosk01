"""
test_lab_tabular_pipeline.py
============================
Validation test suite for the 3-stage memory-efficient tabular lab report extraction pipeline:
1. LabPreprocessor (CPU OpenCV deskew, horizontal gutter detection, adaptive strip slicing)
2. ConstrainedVLMExtractor (Strict JSON schema enforcement, simulated/real VLM ingestion)
3. LabReportPipeline (Boundary deduplication, canonicalization, and reference range validation)
"""

import os
import sys
import unittest
import numpy as np
import cv2

from perception.lab.lab_preprocessor import LabPreprocessor, LabImageChunk
from perception.lab.vlm_extractor import ConstrainedVLMExtractor, LabTestItem, FlagEnum
from perception.lab.lab_pipeline import LabReportPipeline, LabReportResult


def create_synthetic_tabular_report(width=1600, height=2200) -> np.ndarray:
    """Generates a realistic multi-row, multi-column clinical lab report image."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Clinic Header
    cv2.putText(img, "METROPOLIS CLINICAL LABORATORIES", (width // 4, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    cv2.putText(img, "COMPREHENSIVE HEMATOLOGY & METABOLIC REPORT", (width // 4 - 50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.line(img, (100, 200), (width - 100, 200), (50, 50, 50), 2)

    # Table Header Row
    headers = ["Test / Investigation", "Observed Value", "Unit", "Biological Ref Interval", "Flag"]
    col_x = [120, 650, 900, 1100, 1420]
    header_y = 260

    cv2.rectangle(img, (100, 220), (width - 100, 290), (235, 235, 235), -1)
    cv2.rectangle(img, (100, 220), (width - 100, 290), (100, 100, 100), 1)

    for title, x in zip(headers, col_x):
        cv2.putText(img, title, (x, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Rows of data
    rows = [
        ("Hemoglobin (Hb)", "10.4", "g/dL", "13.0 - 17.0", "LOW"),
        ("Total Leukocyte Count (WBC)", "8,200", "/cumm", "4000 - 11000", ""),
        ("RBC Count", "3.9", "mil/uL", "4.5 - 5.5", "LOW"),
        ("Packed Cell Volume (PCV)", "33.2", "%", "36.0 - 50.0", "LOW"),
        ("Platelet Count", "190,000", "/cumm", "150000 - 450000", ""),
        ("Neutrophils", "64", "%", "40 - 70", ""),
        ("Lymphocytes", "28", "%", "20 - 40", ""),
        ("Monocytes", "5", "%", "2 - 10", ""),
        ("Eosinophils", "3", "%", "1 - 6", ""),
        ("Serum Creatinine", "1.65", "mg/dL", "0.7 - 1.2", "HIGH"),
        ("Blood Urea", "48.0", "mg/dL", "15 - 45", "HIGH"),
        ("Total Bilirubin", "0.9", "mg/dL", "0.2 - 1.2", "")
    ]

    current_y = 350
    for row in rows:
        test_name, val, unit, ref, flag = row
        cv2.putText(img, test_name, (col_x[0], current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1)
        cv2.putText(img, val, (col_x[1], current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1)
        cv2.putText(img, unit, (col_x[2], current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1)
        cv2.putText(img, ref, (col_x[3], current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 1)
        if flag:
            color = (0, 0, 200) if flag in ["HIGH", "LOW"] else (10, 10, 10)
            cv2.putText(img, flag, (col_x[4], current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Light divider line between rows
        cv2.line(img, (100, current_y + 25), (width - 100, current_y + 25), (220, 220, 220), 1)
        current_y += 75

    return img


class TestLabTabularPipeline(unittest.TestCase):

    def setUp(self):
        self.preprocessor = LabPreprocessor(
            rows_per_chunk=5,
            overlap_rows=1,
            target_strip_width=1120
        )
        self.extractor = ConstrainedVLMExtractor(backend="mock")
        self.pipeline = LabReportPipeline(
            extractor=self.extractor,
            preprocessor=self.preprocessor
        )
        self.test_img = create_synthetic_tabular_report()

    def test_stage_1_preprocessing_and_slicing(self):
        """Validates deskewing, horizontal gutter detection, and header preservation."""
        deskewed, angle = self.preprocessor.deskew_page(self.test_img)
        self.assertEqual(deskewed.shape[:2], self.test_img.shape[:2])

        chunks = self.preprocessor.slice_table_into_strips(deskewed)
        self.assertGreaterEqual(len(chunks), 1)

        # Verify vision token bounds per chunk (Must be well below 800 tokens to preserve 8GB VRAM)
        for ch in chunks:
            self.assertLessEqual(ch.token_estimated_count, 700)
            self.assertLessEqual(ch.image_bgr.shape[1], self.preprocessor.target_strip_width)
            self.assertLessEqual(ch.image_bgr.shape[0], self.preprocessor.max_chunk_height)

        # Verify header is prepended to subsequent chunks if multiple exist
        if len(chunks) > 1:
            self.assertTrue(chunks[1].has_prepended_header)

    def test_stage_2_constrained_vlm_schema(self):
        """Verifies Pydantic schema validation and JSON sanitization."""
        test_chunk = LabImageChunk(
            chunk_index=0,
            total_chunks=1,
            image_bgr=cv2.resize(self.test_img, (1280, 600)),
            row_start_idx=0,
            row_end_idx=4,
            has_prepended_header=False,
            original_bbox=(0, 600, 0, 1600),
            token_estimated_count=350
        )

        items = self.extractor.extract_chunk(test_chunk)
        self.assertGreaterEqual(len(items), 3)

        for item in items:
            self.assertIsInstance(item.test_name, str)
            self.assertTrue(len(item.test_name) > 0)
            if item.flag:
                self.assertIn(item.flag, [FlagEnum.HIGH, FlagEnum.LOW, FlagEnum.ABNORMAL, FlagEnum.NORMAL])

    def test_stage_3_end_to_end_pipeline(self):
        """Verifies deduplication, range parsing, and clinical summary generation."""
        result = self.pipeline.process(self.test_img, document_id="TEST_CBC_001")

        self.assertIsInstance(result, LabReportResult)
        self.assertGreater(result.total_tests_detected, 0)
        self.assertGreater(result.total_vision_tokens_estimated, 0)

        # Check that abnormal findings are properly identified
        abnormal_names = [a["test_name"] for a in result.flagged_abnormalities]
        self.assertTrue(any("Hemoglobin" in name for name in abnormal_names))

        print(f"\n✅ Lab Report Extraction Pipeline Success:")
        print(f"   Total Analytes Detected: {result.total_tests_detected}")
        print(f"   Abnormal Parameters: {result.abnormal_count}")
        print(f"   Vision Tokens Estimated: {result.total_vision_tokens_estimated}")
        print(f"   Clinical Summary: {result.summary}")


if __name__ == "__main__":
    unittest.main()
