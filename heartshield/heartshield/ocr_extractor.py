"""
HeartShield - OCR Medical Report Extraction
---------------------------------------------
Extracts raw text from uploaded PDF or image medical reports, then
parses out the clinical values HeartShield understands using
regex patterns tolerant of common lab-report phrasing and OCR noise.

Honesty note: a typical patient-facing lab report (CBC / lipid panel /
vitals printout) usually only contains a subset of the 13 clinical
fields the ML model was trained on -- age, resting blood pressure,
cholesterol, fasting blood sugar, and sometimes resting/max heart rate.
Fields that come from a clinical exam or stress test (chest pain type,
exercise-induced angina, ST depression, thalassemia result, number of
vessels) are rarely printed on a routine report and are NOT reliably
extractable by OCR. This module only ever returns fields it is
reasonably confident about; everything else is left for the user to
fill in manually, and the calling app must always show extracted
values to the user for confirmation before prediction -- OCR values
are a convenience prefill, never an unreviewed input to the model.
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader


@dataclass
class ExtractionResult:
    raw_text: str
    fields: dict = field(default_factory=dict)
    confidence_notes: list = field(default_factory=list)


def extract_text_from_pdf(path: str) -> str:
    """Try native text extraction first (fast, accurate for digital PDFs);
    fall back to OCR via page rasterization for scanned/image-only PDFs."""
    text_chunks = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text() or ""
            text_chunks.append(t)
    except Exception:
        pass

    native_text = "\n".join(text_chunks).strip()
    if len(native_text) > 40:  # looks like a real digital PDF
        return native_text

    # Fall back to OCR on rasterized pages (scanned report)
    ocr_chunks = []
    try:
        pages = convert_from_path(path, dpi=300)
        for page_img in pages:
            ocr_chunks.append(pytesseract.image_to_string(page_img))
    except Exception as e:
        ocr_chunks.append("")
    return "\n".join(ocr_chunks).strip()


def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img)


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------
# Regex parsing of clinical values from noisy OCR/report text
# ---------------------------------------------------------------------

_NUM = r"(\d{1,3}(?:\.\d+)?)"

PATTERNS = {
    "age": [
        rf"\bage\b[:\s]*{_NUM}",
    ],
    "sex": [
        r"\bsex\b[:\s]*\b(male|female|m|f)\b",
        r"\bgender\b[:\s]*\b(male|female|m|f)\b",
    ],
    "trestbps": [  # resting systolic blood pressure
        rf"\bblood\s*pressure\b[:\s]*{_NUM}\s*/\s*\d{{1,3}}",
        rf"\bBP\b[:\s]*{_NUM}\s*/\s*\d{{1,3}}",
        rf"\bsystolic\b[:\s]*{_NUM}",
        rf"\bresting\s*bp\b[:\s]*{_NUM}",
    ],
    "chol": [
        rf"\btotal\s*cholesterol\b[:\s]*{_NUM}",
        rf"\bserum\s*cholesterol\b[:\s]*{_NUM}",
        rf"\bcholesterol\b[:\s]*{_NUM}\s*mg\s*/\s*dl",
        rf"\bcholesterol\b[:\s]*{_NUM}",
    ],
    "fbs_value": [  # raw glucose mg/dl -> converted to the fbs>120 flag later
        rf"\bfasting\s*blood\s*sugar\b[:\s]*{_NUM}",
        rf"\bfasting\s*glucose\b[:\s]*{_NUM}",
        rf"\bglucose\s*\(fasting\)\b[:\s]*{_NUM}",
        rf"\bFBS\b[:\s]*{_NUM}",
    ],
    "thalach": [  # max / resting heart rate
        rf"\bmax(?:imum)?\s*heart\s*rate\b[:\s]*{_NUM}",
        rf"\bheart\s*rate\b[:\s]*{_NUM}\s*bpm",
        rf"\bpulse\s*rate\b[:\s]*{_NUM}",
        rf"\bHR\b[:\s]*{_NUM}\s*bpm",
    ],
}

_SEX_MAP = {"male": 1, "m": 1, "female": 0, "f": 0}


def parse_fields(text: str) -> ExtractionResult:
    lowered = text.lower()
    fields = {}
    notes = []

    for field_name, patterns in PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, lowered, flags=re.IGNORECASE)
            if m:
                value = m.group(1)
                if field_name == "sex":
                    fields["sex"] = _SEX_MAP.get(value.lower())
                else:
                    try:
                        fields[field_name] = float(value)
                    except ValueError:
                        continue
                break  # stop at first matching pattern for this field

    # Derive the categorical fasting-blood-sugar flag the model expects
    if "fbs_value" in fields:
        fields["fbs"] = 1 if fields["fbs_value"] > 120 else 0

    if not fields:
        notes.append(
            "No recognizable clinical values were found in this document. "
            "You can still enter your health details manually below."
        )
    else:
        found = ", ".join(sorted(fields.keys()))
        notes.append(f"Auto-extracted from report: {found}. Please review and correct before predicting.")

    unfound_core = [f for f in ["age", "sex", "trestbps", "chol", "fbs", "thalach"] if f not in fields]
    if unfound_core:
        notes.append(
            "Not found in the report (please fill in manually): " + ", ".join(unfound_core)
        )

    notes.append(
        "Note: exam-derived fields (chest pain type, exercise angina, ST depression, "
        "vessel count, thalassemia result) generally aren't printed on routine lab "
        "reports and always require manual entry."
    )

    return ExtractionResult(raw_text=text, fields=fields, confidence_notes=notes)


def extract_from_file(file_path: str) -> ExtractionResult:
    text = extract_text(file_path)
    return parse_fields(text)
