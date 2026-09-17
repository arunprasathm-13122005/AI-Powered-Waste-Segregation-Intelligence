import streamlit as st
from ultralytics import YOLO
import threading
import time
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import av
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
    )
    try:
        from streamlit_autorefresh import st_autorefresh
        LIVE_REFRESH_AVAILABLE = True
    except ImportError:
        LIVE_REFRESH_AVAILABLE = False
    LIVE_CAMERA_AVAILABLE = True
except ImportError:
    LIVE_CAMERA_AVAILABLE = False
    LIVE_REFRESH_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Powered Waste Segregation Intelligence",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #f7f9f8;
}

.block-container {
    max-width: 1180px;
    padding: 2rem 2rem 3rem;
}

section[data-testid="stSidebar"] { display: none; }

/* Classic typography */
.page-title {
    color: #14532d;
    font-size: 38px;
    font-weight: 800;
    margin: 0;
}
.page-subtitle {
    color: #5b6870;
    font-size: 15px;
    margin-top: 4px;
    margin-bottom: 30px;
}
.section-title {
    color: #14532d;
    font-size: 25px;
    font-weight: 750;
    margin: 22px 0 8px;
}
.section-caption {
    color: #68747c;
    font-size: 14px;
    margin-bottom: 14px;
}

/* Simple scanner header */
.scanner-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 4px;
}
.scanner-status {
    color: #166534;
    background: #ecfdf3;
    border: 1px solid #bbf7d0;
    border-radius: 7px;
    padding: 7px 11px;
    font-size: 12px;
    font-weight: 600;
}

/* Classic radio selector */
div[data-testid="stRadio"] > label {
    font-size: 14px !important;
    color: #334155 !important;
}
div[data-testid="stRadio"] > div {
    gap: 18px !important;
}

/* Standard cards */
.classic-card {
    background: #ffffff;
    border: 1px solid #cfd8d3;
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0 18px;
}
.card-heading {
    color: #1f2937;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 4px;
}
.card-description {
    color: #64748b;
    font-size: 13px;
    line-height: 1.55;
}

/* Camera */
.camera-card {
    background: #ffffff;
    border: 1px solid #cfd8d3;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
}
.camera-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 12px;
    margin-bottom: 12px;
}
.camera-title {
    color: #374151;
    font-size: 16px;
    font-weight: 700;
}
.camera-description {
    color: #6b7280;
    font-size: 13px;
    margin-top: 3px;
}
.camera-status {
    color: #166534;
    font-size: 12px;
    font-weight: 600;
}
.camera-help {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e5a9e;
    border-radius: 8px;
    padding: 11px 13px;
    margin-top: 12px;
    font-size: 13px;
    line-height: 1.5;
}

/* Confidence */
.confidence-card {
    background: #ffffff;
    border: 1px solid #d6ded9;
    border-radius: 9px;
    padding: 14px 17px 8px;
    margin: 12px 0 18px;
}
.confidence-label {
    color: #374151;
    font-size: 13px;
    font-weight: 650;
}

/* Native controls */
.stButton > button {
    border-radius: 7px !important;
}

div[data-testid="stFileUploader"] section {
    background: #f8fafc !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 12px;
}

@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .page-title { font-size: 30px; }
    .scanner-header { align-items: flex-start; flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOUR TRAINED MODEL
# ============================================================

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "yolo_saved.pt"

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# ============================================================
# EXPLICIT WASTE SEGREGATION MAP
#
# IMPORTANT:
# YOLO CLASS != SEGREGATION CATEGORY
#
# YOLO detects the object.
# This mapping decides the waste stream.
# ============================================================

WASTE_MAP = {

    # --------------------------------------------------------
    # HAZARDOUS
    # --------------------------------------------------------

    "Battery": {
        "category": "Hazardous",
        "degradable": "Non-Degradable",
        "recyclable": "Specialized Recycling",
        "destination": "Hazardous / E-Waste Collection",
        "action": "Keep separate. Do not place in normal waste bins."
    },

    "Aerosol": {
        "category": "Hazardous",
        "degradable": "Non-Degradable",
        "recyclable": "Specialized Recycling",
        "destination": "Hazardous Waste Collection",
        "action": "Keep separate and send to an authorized collection facility."
    },


    # --------------------------------------------------------
    # ORGANIC
    # --------------------------------------------------------

    "Food waste": {
        "category": "Organic",
        "degradable": "Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "Organic / Compost Bin",
        "action": "Send for composting or organic waste processing."
    },


    # --------------------------------------------------------
    # GENERAL / NON-RECYCLABLE
    # --------------------------------------------------------

    "Foam cup": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Send to general waste processing."
    },

    "Foam food container": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Send to general waste processing."
    },

    "Styrofoam piece": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Send to general waste processing."
    },

    "Shoe": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General / Textile Collection",
        "action": "Reuse or send to an appropriate textile/shoe collection facility."
    },

    "Tissues": {
        "category": "General",
        "degradable": "Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Do not mix with recyclable paper."
    },

    "Cigarette": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Dispose of safely in general waste."
    },


    # --------------------------------------------------------
    # RECYCLABLE - METAL
    # --------------------------------------------------------

    "Aluminium foil": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Clean and send to metal recycling where accepted."
    },

    "Aluminium blister pack": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal / Blister Recycling",
        "action": "Send to an appropriate recycling stream."
    },

    "Carded blister pack": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Material Recovery Facility",
        "action": "Separate materials where possible before recycling."
    },

    "Food Can": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Empty and clean before recycling."
    },

    "Drink can": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Empty and send to metal recycling."
    },

    "Metal bottle cap": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Collect with compatible metal recycling."
    },

    "Metal lid": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Send to metal recycling."
    },

    "Pop tab": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Collect with aluminium/metal recycling."
    },

    "Scrap metal": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Metal Recycling",
        "action": "Send to a scrap-metal recycler."
    },


    # --------------------------------------------------------
    # RECYCLABLE - GLASS
    # --------------------------------------------------------

    "Glass bottle": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Glass Recycling",
        "action": "Separate from other waste and send for glass recycling."
    },

    "Broken glass": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Glass Recycling",
        "action": "Handle carefully and place in a designated glass stream."
    },

    "Glass cup": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Glass Recycling",
        "action": "Send to glass recycling where accepted."
    },

    "Glass jar": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Glass Recycling",
        "action": "Empty and clean before recycling."
    },


    # --------------------------------------------------------
    # RECYCLABLE - PLASTIC
    # --------------------------------------------------------

    "Other plastic bottle": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Empty, clean and send to plastic recycling."
    },

    "Clear plastic bottle": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Empty and send to plastic recycling."
    },

    "Plastic bottle cap": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Collect separately or with compatible plastic recycling."
    },

    "Plastic lid": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Send to plastic recycling."
    },

    "Other plastic": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Material Recovery",
        "action": "Send to a facility that accepts the identified plastic."
    },

    "Disposable plastic cup": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Clean and send to an appropriate plastic recycling stream."
    },

    "Other plastic cup": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Clean before recycling."
    },

    "Plastic film": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Flexible Plastic Collection",
        "action": "Send to a facility that accepts flexible plastic."
    },

    "Six pack rings": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Collect and send to an appropriate plastic recycler."
    },

    "Garbage bag": {
        "category": "Manual Review",
        "degradable": "Unknown",
        "recyclable": "Unknown",
        "destination": "Secondary Inspection",
        "action": "Opaque bag detected. Inspect the bag contents before automatic segregation."
    },

    "Other plastic wrapper": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Flexible Plastic Collection",
        "action": "Send to an appropriate flexible-plastic facility."
    },

    "Single-use carrier bag": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Reuse where possible or recycle where accepted."
    },

    "Polypropylene bag": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Send to a facility accepting polypropylene."
    },

    "Spread tub": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Empty and clean before recycling."
    },

    "Tupperware": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Reuse where possible or recycle according to local acceptance."
    },

    "Disposable food container": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic / Material Recovery",
        "action": "Clean and send to an appropriate recycling stream."
    },

    "Other plastic container": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Plastic Recycling",
        "action": "Clean and send to plastic recycling."
    },

    "Plastic glooves": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Dispose of used gloves as general waste."
    },

    "Plastic utensils": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Dispose as general waste unless a local recycling program accepts them."
    },

    "Plastic straw": {
        "category": "General",
        "degradable": "Non-Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General Waste Bin",
        "action": "Dispose as general waste unless specifically accepted for recycling."
    },


    # --------------------------------------------------------
    # RECYCLABLE - PAPER / CARTON
    # --------------------------------------------------------

    "Toilet tube": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper Recycling",
        "action": "Keep dry and send with recyclable paper/cardboard."
    },

    "Other carton": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper / Cardboard Recycling",
        "action": "Flatten and keep dry before recycling."
    },

    "Egg carton": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper / Cardboard Recycling",
        "action": "Keep dry and send for paper recycling."
    },

    "Drink carton": {
        "category": "Recyclable",
        "degradable": "Non-Degradable",
        "recyclable": "Recyclable",
        "destination": "Carton Recycling",
        "action": "Empty and send to a facility accepting beverage cartons."
    },

    "Corrugated carton": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Cardboard Recycling",
        "action": "Flatten and keep dry."
    },

    "Meal carton": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper / Cardboard Recycling",
        "action": "Remove food residue where possible before recycling."
    },

    "Pizza box": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper / Cardboard Recycling",
        "action": "Recycle clean portions; heavily food-soiled portions may require separate handling."
    },

    "Magazine paper": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper Recycling",
        "action": "Keep dry and send to paper recycling."
    },

    "Wrapping paper": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper Recycling",
        "action": "Recycle if free from plastic/foil coating."
    },

    "Normal paper": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper Recycling",
        "action": "Keep dry and clean."
    },

    "Paper bag": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Paper Recycling",
        "action": "Reuse or send to paper recycling."
    },

    "Paper cup": {
        "category": "Recyclable",
        "degradable": "Degradable",
        "recyclable": "Recyclable",
        "destination": "Cup / Paper Recycling",
        "action": "Recycle only where the facility accepts coated paper cups."
    },

    "Paper straw": {
        "category": "General",
        "degradable": "Degradable",
        "recyclable": "Non-Recyclable",
        "destination": "General / Organic Processing",
        "action": "Dispose according to local paper-waste acceptance."
    },


    # --------------------------------------------------------
    # OTHER MATERIALS
    # --------------------------------------------------------

    "Plastified paper bag": {
        "category": "Manual Review",
        "degradable": "Non-Degradable",
        "recyclable": "Conditional",
        "destination": "Material Recovery Facility",
        "action": "Check whether the local facility accepts laminated paper."
    },

    "Crisp packet": {
        "category": "Manual Review",
        "degradable": "Non-Degradable",
        "recyclable": "Conditional",
        "destination": "Flexible Packaging Collection",
        "action": "Send only to a facility that accepts multilayer snack packaging."
    },

    "Rope & strings": {
        "category": "Manual Review",
        "degradable": "Non-Degradable",
        "recyclable": "Conditional",
        "destination": "Material Recovery / Specialized Collection",
        "action": "Separate from normal recycling because long fibres can interfere with sorting equipment."
    },

    "Squeezable tube": {
        "category": "Manual Review",
        "degradable": "Non-Degradable",
        "recyclable": "Conditional",
        "destination": "Specialized Plastic Collection",
        "action": "Recycle only where the material is accepted."
    },

    "Unlabeled litter": {
        "category": "Manual Review",
        "degradable": "Unknown",
        "recyclable": "Unknown",
        "destination": "Manual Inspection",
        "action": "Human verification is required before sorting."
    }
}


# ============================================================
# FALLBACK
# ============================================================

DEFAULT_WASTE_INFO = {
    "category": "Manual Review",
    "degradable": "Unknown",
    "recyclable": "Unknown",
    "destination": "Manual Inspection",
    "action": "Object class detected, but no segregation rule is configured."
}


# ============================================================
# GET WASTE INFORMATION
# ============================================================

def get_waste_information(class_name):

    return WASTE_MAP.get(
        class_name,
        DEFAULT_WASTE_INFO
    )


# ============================================================
# CONFIDENCE DECISION
# ============================================================

def get_decision(confidence):

    if confidence >= 0.80:
        return "AUTO SORT"

    elif confidence >= 0.50:
        return "VERIFY"

    else:
        return "MANUAL REVIEW"


# ============================================================
# ICON
# ============================================================

def get_icon(category):

    icons = {
        "Recyclable": "♻️",
        "Organic": "🌱",
        "Hazardous": "⚠️",
        "General": "🗑️",
        "Manual Review": "🔎"
    }

    return icons.get(category, "❓")


# ============================================================
# RESULT BOX CLASS
# ============================================================

def get_box_class(category):

    if category == "Recyclable":
        return "recycle-box"

    if category == "Organic":
        return "organic-box"

    if category == "Hazardous":
        return "hazard-box"

    if category == "General":
        return "general-box"

    return "review-box"


# ============================================================
# HIDDEN / OPAQUE BAG CHECK
# ============================================================

def get_hidden_content_status(class_name, confidence):
    """Flag opaque bags instead of guessing what is inside them."""
    name = str(class_name).strip().lower()
    if name in {"garbage bag", "black carry bag", "carry bag"}:
        if confidence >= 0.80:
            return "HIGH", "Opaque bag detected — secondary inspection required."
        return "POSSIBLE", "Possible opaque bag — verify contents before sorting."
    return "LOW", "No opaque-container warning."


def detect_dark_region(image_rgb):
    """Conservative visual warning; it does not identify hidden objects."""
    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[2] < 3:
        return False, 0.0
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
    dark = (hsv[:, :, 2] < 55) & (hsv[:, :, 1] < 100)
    ratio = float(np.mean(dark))
    return ratio >= 0.30, ratio


# ============================================================
# LIVE CAMERA SCANNER
# ============================================================

if LIVE_CAMERA_AVAILABLE:

    class WasteCameraProcessor(VideoProcessorBase):
        """
        Clean live YOLOv8 camera processor.

        The camera frame is kept as close as possible to the original
        DroidCam/webcam image. YOLO annotations are added only when an
        object is detected. No artificial green/black scanner background
        or large text panel is drawn over the video.
        """

        confidence_value = 0.30

        def __init__(self):
            self.model = load_model()
            self.confidence = float(WasteCameraProcessor.confidence_value)
            self._lock = threading.Lock()
            self.latest_detections = []
            self.latest_hidden_risk = "LOW"
            self.latest_status = "Waiting for camera frames..."
            self.latest_timestamp = 0.0

        def get_live_snapshot(self):
            """Return the latest live YOLO analysis safely to the Streamlit thread."""
            with self._lock:
                return {
                    "detections": [dict(item) for item in self.latest_detections],
                    "hidden_risk": self.latest_hidden_risk,
                    "status": self.latest_status,
                    "timestamp": self.latest_timestamp,
                }

        def recv(self, frame):
            try:
                # Read the WebRTC frame directly as BGR.
                # This avoids unnecessary RGB <-> BGR conversions and
                # works better with many virtual webcam drivers such as DroidCam.
                try:
                    frame_bgr = frame.to_ndarray(format="bgr24")
                except Exception:
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    frame_bgr = cv2.cvtColor(
                        frame_rgb,
                        cv2.COLOR_RGB2BGR
                    )

                # Keep processing light enough for live camera use.
                h, w = frame_bgr.shape[:2]

                max_width = 960
                if w > max_width:
                    scale = max_width / float(w)
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (
                            int(w * scale),
                            int(h * scale)
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                # Keep the Streamlit slider value synchronized with the
                # persistent WebRTC processor.
                self.confidence = float(WasteCameraProcessor.confidence_value)

                # YOLOv8 inference
                results = self.model.predict(
                    source=frame_bgr,
                    conf=float(self.confidence),
                    imgsz=640,
                    verbose=False
                )

                result = results[0]

                has_detection = (
                    result.boxes is not None
                    and len(result.boxes) > 0
                )

                # ----------------------------------------------------
                # IMPORTANT:
                # If nothing is detected, return the original camera
                # image. This gives a natural webcam view.
                # ----------------------------------------------------
                if not has_detection:
                    with self._lock:
                        self.latest_detections = []
                        self.latest_hidden_risk = "LOW"
                        self.latest_status = "No waste object detected above the current threshold."
                        self.latest_timestamp = time.time()

                    return av.VideoFrame.from_ndarray(
                        frame_bgr,
                        format="bgr24"
                    )

                # YOLO boxes + labels
                annotated_bgr = result.plot(
                    labels=True,
                    boxes=True,
                    conf=True
                )

                class_names = result.names
                hidden_detected = False

                # Check opaque/hidden containers
                for class_id, confidence in zip(
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()
                ):
                    class_id = int(class_id)
                    confidence = float(confidence)

                    if isinstance(class_names, dict):
                        class_name = class_names.get(
                            class_id,
                            f"Unknown Class {class_id}"
                        )
                    else:
                        class_name = class_names[class_id]

                    hidden_risk, _ = get_hidden_content_status(
                        class_name,
                        confidence
                    )

                    if hidden_risk in {"HIGH", "POSSIBLE"}:
                        hidden_detected = True
                        break

                # ----------------------------------------------------
                # Build detailed live segregation results
                # ----------------------------------------------------
                live_detections = []
                hidden_risks = []

                for class_id, confidence in zip(
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()
                ):
                    class_id = int(class_id)
                    confidence = float(confidence)

                    if isinstance(class_names, dict):
                        class_name = class_names.get(
                            class_id,
                            f"Unknown Class {class_id}"
                        )
                    else:
                        class_name = class_names[class_id]

                    waste_info = get_waste_information(class_name)
                    category = waste_info["category"]
                    hidden_risk, hidden_message = get_hidden_content_status(
                        class_name,
                        confidence
                    )

                    confidence_decision = get_decision(confidence)
                    decision = (
                        "MANUAL REVIEW"
                        if hidden_risk in {"HIGH", "POSSIBLE"}
                        else confidence_decision
                    )

                    live_detections.append({
                        "Object": class_name,
                        "Confidence": round(confidence * 100, 1),
                        "Waste Stream": category,
                        "Degradable": waste_info["degradable"],
                        "Recyclable": waste_info["recyclable"],
                        "Destination": waste_info["destination"],
                        "Decision": decision,
                        "Recommendation": waste_info["action"],
                        "Hidden Risk": hidden_risk,
                        "Inspection Status": hidden_message,
                    })

                    hidden_risks.append(hidden_risk)

                object_count = len(live_detections)

                with self._lock:
                    self.latest_detections = live_detections
                    self.latest_hidden_risk = (
                        "HIGH" if "HIGH" in hidden_risks
                        else "POSSIBLE" if "POSSIBLE" in hidden_risks
                        else "LOW"
                    )
                    self.latest_status = (
                        f"{object_count} object(s) detected and classified."
                    )
                    self.latest_timestamp = time.time()

                # ----------------------------------------------------
                # Small, clean status badge
                # ----------------------------------------------------

                badge_text = f"LIVE  |  {object_count} object"
                if object_count != 1:
                    badge_text += "s"

                # Small dark badge, not a full-width overlay.
                cv2.rectangle(
                    annotated_bgr,
                    (12, 12),
                    (205, 47),
                    (35, 35, 35),
                    -1
                )

                cv2.putText(
                    annotated_bgr,
                    badge_text,
                    (23, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # ----------------------------------------------------
                # Opaque bag warning
                # ----------------------------------------------------
                if hidden_detected:
                    h2, w2 = annotated_bgr.shape[:2]

                    warning_text = (
                        "OPAQUE BAG - SECONDARY INSPECTION"
                    )

                    text_size = cv2.getTextSize(
                        warning_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2
                    )[0]

                    warning_w = min(
                        text_size[0] + 28,
                        w2 - 20
                    )

                    warning_x = 10
                    warning_y = h2 - 50

                    cv2.rectangle(
                        annotated_bgr,
                        (warning_x, warning_y),
                        (
                            warning_x + warning_w,
                            h2 - 12
                        ),
                        (45, 45, 190),
                        -1
                    )

                    cv2.putText(
                        annotated_bgr,
                        warning_text,
                        (warning_x + 12, h2 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                return av.VideoFrame.from_ndarray(
                    annotated_bgr,
                    format="bgr24"
                )

            except Exception as e:
                with self._lock:
                    self.latest_status = f"Live inference error: {str(e)[:120]}"
                    self.latest_timestamp = time.time()

                # If YOLO processing fails, show the real camera image
                # instead of replacing it with a fake coloured screen.
                try:
                    try:
                        fallback_bgr = frame.to_ndarray(format="bgr24")
                    except Exception:
                        fallback_rgb = frame.to_ndarray(format="rgb24")
                        fallback_bgr = cv2.cvtColor(
                            fallback_rgb,
                            cv2.COLOR_RGB2BGR
                        )

                    return av.VideoFrame.from_ndarray(
                        fallback_bgr,
                        format="bgr24"
                    )

                except Exception:
                    return frame


# ============================================================
# CLASSIC HEADER
# ============================================================

st.markdown(
    """
    <div class="page-title">♻️ Waste Intelligence</div>
    <div class="page-subtitle">YOLOv8-based waste detection, intelligent segregation and monitoring system.</div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# WASTE SCANNER
# ============================================================

st.markdown(
    """
    <div class="scanner-header">
        <div>
            <div class="section-title">📷 Waste Scanner</div>
            <div class="section-caption">Select a scanning method to analyse waste.</div>
        </div>
        <div class="scanner-status">● AI Model Ready</div>
    </div>
    """,
    unsafe_allow_html=True
)

scan_mode = st.radio(
    "Select scanning method",
    ["📁 Image Upload", "📹 Live Camera"],
    horizontal=True,
    key="scan_mode"
)

uploaded_file = None
confidence_threshold = 0.30

# ============================================================
# IMAGE UPLOAD
# ============================================================

if scan_mode == "📁 Image Upload":

    st.markdown(
        """
        <div class="classic-card">
            <div class="card-heading">📁 Image Upload</div>
            <div class="card-description">
                Upload a saved waste image for YOLOv8 analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing one or more waste objects."
    )

    st.markdown('<div class="confidence-card">', unsafe_allow_html=True)
    st.markdown('<div class="confidence-label">🎯 Detection Confidence</div>', unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        key="upload_confidence",
        label_visibility="collapsed"
    )

    st.caption(f"Current detection threshold: {confidence_threshold:.2f} ({confidence_threshold * 100:.0f}%)")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# LIVE CAMERA
# ============================================================

else:

    if not LIVE_CAMERA_AVAILABLE:
        st.error("📹 Live Camera support is not installed.")
        st.code("python -m pip install streamlit-webrtc av")
        st.stop()

    st.markdown(
        """
        <div class="camera-card">
            <div class="camera-header">
                <div>
                    <div class="camera-title">📹 Live Laptop Camera</div>
                    <div class="camera-description">
                        Scan waste continuously using your laptop webcam. No photo capture is required.
                    </div>
                </div>
                <div class="camera-status">● Camera Scanner</div>
            </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="confidence-card">', unsafe_allow_html=True)
    st.markdown('<div class="confidence-label">🎯 Camera Detection Confidence</div>', unsafe_allow_html=True)

    camera_confidence = st.slider(
        "Camera Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        key="camera_confidence",
        label_visibility="collapsed"
    )

    st.caption(f"Current camera threshold: {camera_confidence:.2f} ({camera_confidence * 100:.0f}%)")
    st.markdown('</div>', unsafe_allow_html=True)

    WasteCameraProcessor.confidence_value = camera_confidence

    live_ctx = webrtc_streamer(
        key="waste-live-camera",
        video_processor_factory=WasteCameraProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "max": 1280},
                "height": {"ideal": 720, "max": 720},
                "frameRate": {"ideal": 30, "max": 30}
            },
            "audio": False
        },
        rtc_configuration=RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            }
        ),
        async_processing=True
    )

    st.markdown(
        """
        <div class="camera-help">
            <b>How to use:</b> Click <b>START</b>, allow camera permission, select the required camera if your browser shows more than one device, and place the waste inside the frame. YOLOv8 analyses the live video continuously.
            <br><br>
            <b>Opaque bag:</b> If the waste is completely hidden inside a black/opaque bag, the RGB camera cannot identify the hidden object and the system will flag it for secondary inspection.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # LIVE DETECTION DETAILS
    # ========================================================
    # WebRTC inference runs in a background thread. The small
    # auto-refresh below lets Streamlit display the latest
    # detections without taking a photo or stopping the camera.
    # ========================================================

    if live_ctx.state.playing and LIVE_REFRESH_AVAILABLE:
        st_autorefresh(
            interval=1200,
            limit=None,
            key="live-detection-refresh"
        )

    live_snapshot = None
    if live_ctx.video_processor is not None:
        try:
            live_snapshot = live_ctx.video_processor.get_live_snapshot()
        except Exception:
            live_snapshot = None

    st.markdown(
        '<div class="section-title">🔍 Live Detection Details</div>',
        unsafe_allow_html=True
    )

    if live_snapshot is None:
        st.info("Start the camera to receive live YOLO detection details.")
    else:
        live_df = pd.DataFrame(live_snapshot["detections"])

        if live_df.empty:
            st.info(
                "📷 Camera is running. No waste object is currently "
                "detected above the selected confidence threshold."
            )
        else:
            live_total = len(live_df)
            live_recyclable = int(
                (live_df["Waste Stream"] == "Recyclable").sum()
            )
            live_organic = int(
                (live_df["Waste Stream"] == "Organic").sum()
            )
            live_hazardous = int(
                (live_df["Waste Stream"] == "Hazardous").sum()
            )
            live_general = int(
                (live_df["Waste Stream"] == "General").sum()
            )
            live_review = int(
                (live_df["Waste Stream"] == "Manual Review").sum()
            )
            live_hidden = int(
                live_df["Hidden Risk"].isin(["HIGH", "POSSIBLE"]).sum()
            )

            # Current live summary
            l1, l2, l3, l4, l5, l6 = st.columns(6)

            with l1:
                st.metric("Objects", live_total)
            with l2:
                st.metric("♻️ Recyclable", live_recyclable)
            with l3:
                st.metric("🌱 Organic", live_organic)
            with l4:
                st.metric("⚠️ Hazardous", live_hazardous)
            with l5:
                st.metric("🔎 Manual Review", live_review)
            with l6:
                st.metric("👜 Hidden Risk", live_hidden)

            # Detailed table
            live_display = live_df.copy()
            live_display["Waste Stream"] = live_display[
                "Waste Stream"
            ].apply(lambda x: f"{get_icon(x)} {x}")

            st.markdown(
                "### 🧠 Intelligent Live Segregation Results"
            )

            st.dataframe(
                live_display,
                use_container_width=True,
                hide_index=True
            )

            # Sorting status
            live_auto = int(
                (live_df["Decision"] == "AUTO SORT").sum()
            )
            live_verify = int(
                (live_df["Decision"] == "VERIFY").sum()
            )
            live_manual = int(
                (live_df["Decision"] == "MANUAL REVIEW").sum()
            )

            st.markdown("### ⚙️ Live Sorting Decision Status")

            d1, d2, d3 = st.columns(3)

            with d1:
                st.metric("🟢 Auto Sort", live_auto)

            with d2:
                st.metric("🟡 Verify", live_verify)

            with d3:
                st.metric("🟣 Manual Review", live_manual)

            # Inspection / hidden-content status
            if live_snapshot["hidden_risk"] in {"HIGH", "POSSIBLE"}:
                st.warning(
                    "👜 Opaque/hidden container detected. "
                    "The RGB camera cannot identify contents that are "
                    "completely hidden. Secondary inspection is required."
                )

            # Current confidence
            avg_live_conf = float(live_df["Confidence"].mean())

            st.markdown(
                f"""
                <div class="classic-card">
                    <div class="card-heading">🎯 Live AI Confidence</div>
                    <div class="card-description">
                        Current average detection confidence:
                        <b>{avg_live_conf:.1f}%</b><br>
                        {live_snapshot["status"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# DECISION LOGIC
# ============================================================

# ============================================================

with st.expander("🧠 AI Decision Logic"):

    st.markdown("""
    **Detection confidence is different from recyclability.**

    **80% – 100%**
    → 🟢 AUTO SORT

    **50% – 79%**
    → 🟡 VERIFY

    **Below 50%**
    → 🟣 MANUAL REVIEW

    The YOLO model identifies the object.  
    The segregation rule engine determines its waste stream.
    """)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("❌ YOLO model could not be loaded.")

    st.code(str(e))

    st.info(
        "Make sure yolo_saved.pt is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error("❌ Could not read the uploaded image.")

        st.code(str(e))

        st.stop()


    # --------------------------------------------------------
    # PIL → OPENCV
    # --------------------------------------------------------

    image_np = np.array(image)

    image_cv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    with st.spinner("🔍 AI is analyzing the image..."):

        results = model.predict(
            source=image_cv,
            conf=confidence_threshold,
            verbose=False
        )


    result = results[0]


    # ========================================================
    # DRAW YOLO RESULTS
    # ========================================================

    output_image = result.plot()

    output_image = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # GET MODEL CLASS NAMES
    # ========================================================

    class_names = result.names


    # ========================================================
    # EXTRACT DETECTIONS
    # ========================================================

    detections = []


    if result.boxes is not None and len(result.boxes) > 0:

        class_ids = result.boxes.cls.cpu().numpy()

        confidences = result.boxes.conf.cpu().numpy()


        for class_id, confidence in zip(
            class_ids,
            confidences
        ):

            class_id = int(class_id)

            confidence = float(confidence)


            # ------------------------------------------------
            # YOLO CLASS NAME
            # ------------------------------------------------

            if isinstance(class_names, dict):

                class_name = class_names.get(
                    class_id,
                    f"Unknown Class {class_id}"
                )

            else:

                class_name = class_names[class_id]


            # ------------------------------------------------
            # SEGREGATION RULE
            # ------------------------------------------------

            waste_info = get_waste_information(
                class_name
            )


            category = waste_info["category"]

            degradable = waste_info["degradable"]

            recyclable = waste_info["recyclable"]

            destination = waste_info["destination"]

            action = waste_info["action"]

            # ------------------------------------------------
            # HIDDEN / OPAQUE BAG CHECK
            # ------------------------------------------------
            hidden_risk, hidden_message = get_hidden_content_status(
                class_name, confidence
            )

            confidence_decision = get_decision(confidence)

            # Never auto-sort an opaque bag: its contents may be hidden.
            decision = (
                "MANUAL REVIEW"
                if hidden_risk in {"HIGH", "POSSIBLE"}
                else confidence_decision
            )


            # ------------------------------------------------
            # SAVE DETECTION
            # ------------------------------------------------

            detections.append({

                "Object": class_name,

                "Confidence": round(
                    confidence * 100,
                    1
                ),

                "Waste Stream": category,

                "Degradable": degradable,

                "Recyclable": recyclable,

                "Destination": destination,

                "Decision": decision,

                "Recommendation": action,

                "Hidden Risk": hidden_risk,

                "Inspection Status": hidden_message

            })


    # ========================================================
    # DISPLAY IMAGES
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Detection Results</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )


    with col2:

        st.image(
            output_image,
            caption="YOLOv8 Detection",
            use_container_width=True
        )


    # ========================================================
    # OPAQUE BAG / HIDDEN CONTENT WARNING
    # ========================================================

    dark_region_found, dark_region_ratio = detect_dark_region(image)

    garbage_bag_detected = any(
        str(d["Object"]).strip().lower()
        in {"garbage bag", "black carry bag", "carry bag"}
        for d in detections
    )

    if garbage_bag_detected:
        st.warning(
            "👜 **Opaque bag detected. A normal RGB camera cannot identify "
            "objects completely hidden inside it.** This item is routed to "
            "**Secondary Inspection / Manual Review**."
        )
    elif dark_region_found and len(detections) == 0:
        st.warning(
            "🔎 **Possible dark/opaque container detected, but no waste object "
            "is confidently visible.** Manual inspection is recommended."
        )

    # ========================================================
    # NO OBJECTS
    # ========================================================

    if len(detections) == 0:

        st.warning(
            "⚠️ No objects detected above the selected confidence."
        )

        st.info(
            "Try reducing the Detection Confidence slider."
        )

        st.stop()


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        detections
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Waste Segregation Summary</div>',
        unsafe_allow_html=True
    )


    total = len(df)

    recyclable_count = len(
        df[df["Waste Stream"] == "Recyclable"]
    )

    organic_count = len(
        df[df["Waste Stream"] == "Organic"]
    )

    hazardous_count = len(
        df[df["Waste Stream"] == "Hazardous"]
    )

    general_count = len(
        df[df["Waste Stream"] == "General"]
    )

    review_count = len(
        df[df["Waste Stream"] == "Manual Review"]
    )

    hidden_count = len(
        df[df["Hidden Risk"].isin(["HIGH", "POSSIBLE"])]
    )


    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4, c5, c6 = st.columns(6)


    with c1:

        st.metric(
            "Objects",
            total
        )


    with c2:

        st.metric(
            "♻️ Recyclable",
            recyclable_count
        )


    with c3:

        st.metric(
            "🌱 Organic",
            organic_count
        )


    with c4:

        st.metric(
            "⚠️ Hazardous",
            hazardous_count
        )


    with c5:

        st.metric(
            "🔎 Manual Review",
            review_count
        )

    with c6:

        st.metric(
            "👜 Hidden Risk",
            hidden_count
        )


    # ========================================================
    # DETECTED OBJECT TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 Intelligent Segregation Results</div>',
        unsafe_allow_html=True
    )


    display_df = df.copy()


    display_df["Waste Stream"] = display_df[
        "Waste Stream"
    ].apply(
        lambda x: f"{get_icon(x)} {x}"
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # SORTING DECISION
    # ========================================================

    st.markdown(
        '<div class="section-title">⚙️ Sorting Decision Status</div>',
        unsafe_allow_html=True
    )


    auto_count = len(
        df[df["Decision"] == "AUTO SORT"]
    )

    verify_count = len(
        df[df["Decision"] == "VERIFY"]
    )

    manual_count = len(
        df[df["Decision"] == "MANUAL REVIEW"]
    )


    s1, s2, s3 = st.columns(3)


    with s1:

        st.metric(
            "🟢 Auto Sort",
            auto_count
        )


    with s2:

        st.metric(
            "🟡 Verify",
            verify_count
        )


    with s3:

        st.metric(
            "🟣 Manual Review",
            manual_count
        )


# ============================================================
# STOP BEFORE ANALYTICS WHEN NO IMAGE IS UPLOADED
# ============================================================

if uploaded_file is None:
    st.stop()


# ============================================================
# ANALYTICAL WASTE MONITORING DASHBOARD
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📊 Waste Monitoring & Analytics Dashboard
    </div>

    <div style="
        color:#64748B;
        font-size:15px;
        margin-bottom:20px;
    ">
        AI-powered monitoring of waste composition, detection
        confidence and segregation patterns.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION HISTORY
# ============================================================

if "waste_history" not in st.session_state:
    st.session_state.waste_history = []


# Use uploaded filename + current results to avoid
# adding the same analysis repeatedly during reruns

current_signature = (
    uploaded_file.name if uploaded_file else "",
    total,
    recyclable_count,
    organic_count,
    hazardous_count,
    general_count,
    review_count,
    hidden_count
)


if (
    "last_waste_signature"
    not in st.session_state
    or
    st.session_state.last_waste_signature
    != current_signature
):

    st.session_state.waste_history.append({

        "Analysis":
            len(st.session_state.waste_history) + 1,

        "Recyclable":
            recyclable_count,

        "Organic":
            organic_count,

        "Hazardous":
            hazardous_count,

        "General":
            general_count,

        "Manual Review":
            review_count,

        "Hidden Risk":
            hidden_count,

        "Total":
            total,

        "Confidence":
            round(
                df["Confidence"].mean(),
                1
            )
    })

    st.session_state.last_waste_signature = current_signature


# ============================================================
# HISTORY DATAFRAME
# ============================================================

history_df = pd.DataFrame(
    st.session_state.waste_history
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_objects = int(
    history_df["Total"].sum()
)


total_recyclable = int(
    history_df["Recyclable"].sum()
)


total_organic = int(
    history_df["Organic"].sum()
)


total_hazardous = int(
    history_df["Hazardous"].sum()
)


total_general = int(
    history_df["General"].sum()
)


total_manual = int(
    history_df["Manual Review"].sum()
)


average_confidence = round(
    history_df["Confidence"].mean(),
    1
)


# ============================================================
# CURRENT IMAGE PERCENTAGES
# ============================================================

recyclable_percentage = (
    recyclable_count / total * 100
    if total > 0 else 0
)


organic_percentage = (
    organic_count / total * 100
    if total > 0 else 0
)


hazardous_percentage = (
    hazardous_count / total * 100
    if total > 0 else 0
)


general_percentage = (
    general_count / total * 100
    if total > 0 else 0
)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "🔍 Objects Detected",
        total_objects
    )


with k2:

    st.metric(
        "♻️ Recyclable",
        total_recyclable
    )


with k3:

    st.metric(
        "🌱 Organic",
        total_organic
    )


with k4:

    st.metric(
        "⚠️ Hazardous",
        total_hazardous
    )


with k5:

    st.metric(
        "🎯 Avg Confidence",
        f"{average_confidence}%"
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN ANALYTICS ROW
# ============================================================

left, right = st.columns(
    [1.7, 1]
)


# ============================================================
# WASTE PATTERN LINE CHART
# ============================================================

with left:

    st.markdown(
        "### 📈 Waste Pattern Monitoring"
    )

    if len(history_df) >= 2:

        line_df = history_df[
            [
                "Analysis",
                "Recyclable",
                "Organic",
                "Hazardous",
                "General",
                "Manual Review",
                "Hidden Risk"
            ]
        ].copy()


        line_df = line_df.melt(
            id_vars="Analysis",
            var_name="Waste Stream",
            value_name="Objects"
        )


        fig_line = px.line(
            line_df,
            x="Analysis",
            y="Objects",
            color="Waste Stream",
            markers=True,
            line_shape="spline"
        )

                
        st.plotly_chart(
            fig_line,
            use_container_width=True
        )

    else:

        st.info(
            "📷 Analyze at least 2 images to generate "
            "the waste pattern trend."
        )


# ============================================================
# DONUT CHART
# ============================================================

with right:

    st.markdown(
        "### 🍩 Current Waste Composition"
    )


    composition = pd.DataFrame({
        "Category": [
            "Recyclable",
            "Organic",
            "Hazardous",
            "General",
            "Manual Review",
            "Hidden Risk"
        ],

        "Count": [
            recyclable_count,
            organic_count,
            hazardous_count,
            general_count,
            review_count,
            hidden_count
        ]
    })


    composition = composition[
        composition["Count"] > 0
    ]


    if len(composition) > 0:

        fig_donut = px.pie(
            composition,
            names="Category",
            values="Count",
            hole=0.58
        )


        fig_donut.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            paper_bgcolor="white",
            legend_title="Waste Stream"
        )


        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Objects: %{value}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        )


        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

st.markdown(
    "### 📊 Waste Stream Distribution"
)


distribution_df = pd.DataFrame({

    "Waste Stream": [
        "♻️ Recyclable",
        "🌱 Organic",
        "⚠️ Hazardous",
        "🗑️ General",
        "🔎 Manual Review",
        "👜 Hidden Risk"
    ],

    "Objects": [
        recyclable_count,
        organic_count,
        hazardous_count,
        general_count,
        review_count,
        hidden_count
    ]
})


fig_bar = px.bar(
    distribution_df,
    x="Waste Stream",
    y="Objects",
    text="Objects"
)


fig_bar.update_layout(
    height=350,
    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis_title=None,
    yaxis_title="Number of Objects"
)


fig_bar.update_traces(
    textposition="outside"
)


st.plotly_chart(
    fig_bar,
    use_container_width=True
)


# ============================================================
# CONFIDENCE TREND
# ============================================================

st.markdown(
    "### 🎯 AI Detection Confidence Trend"
)


if len(history_df) >= 2:

    confidence_df = history_df[
        [
            "Analysis",
            "Confidence"
        ]
    ].copy()


    fig_confidence = px.line(
        confidence_df,
        x="Analysis",
        y="Confidence",
        markers=True
    )


    fig_confidence.update_layout(
        height=300,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Analysis Sequence",
        yaxis_title="Confidence (%)",
        yaxis=dict(
            range=[0, 100]
        )
    )


    st.plotly_chart(
        fig_confidence,
        use_container_width=True
    )

else:

    st.info(
        "Confidence trend will appear after multiple analyses."
    )


# ============================================================
# ANALYTICAL INSIGHTS
# ============================================================

st.markdown(
    "### 🧠 Waste Pattern Insights"
)


category_counts = {
    "Recyclable": recyclable_count,
    "Organic": organic_count,
    "Hazardous": hazardous_count,
    "General": general_count,
    "Manual Review": review_count,
    "Hidden Risk": hidden_count
}


dominant_category = max(
    category_counts,
    key=category_counts.get
)


dominant_count = category_counts[
    dominant_category
]


dominant_percentage = (
    dominant_count / total * 100
    if total > 0 else 0
)


i1, i2, i3 = st.columns(3)


# ------------------------------------------------------------
# INSIGHT 1
# ------------------------------------------------------------

with i1:

    st.markdown(
        f"""
        <div class="info-box">

        <h4>📌 Dominant Waste Stream</h4>

        <div style="font-size:26px;font-weight:700;">
            {get_icon(dominant_category)}
            {dominant_category}
        </div>

        <p>
            Represents approximately
            <b>{dominant_percentage:.1f}%</b>
            of the current detected waste.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# INSIGHT 2
# ------------------------------------------------------------

with i2:

    st.markdown(
        f"""
        <div class="recycle-box">

        <h4>♻️ Recyclable Pattern</h4>

        <div style="font-size:26px;font-weight:700;">
            {recyclable_percentage:.1f}%
        </div>

        <p>
            of the current detected objects
            are classified into the recyclable stream.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# INSIGHT 3
# ------------------------------------------------------------

with i3:

    st.markdown(
        f"""
        <div class="review-box">

        <h4>🔎 Review Requirement</h4>

        <div style="font-size:26px;font-weight:700;">
            {review_count}
        </div>

        <p>
            detected objects currently require
            manual review.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION ANALYSIS HISTORY
# ============================================================

st.markdown(
    "### 🕒 Monitoring History"
)


history_display = history_df.copy()


history_display = history_display.rename(
    columns={
        "Analysis": "Analysis",
        "Recyclable": "♻️ Recyclable",
        "Organic": "🌱 Organic",
        "Hazardous": "⚠️ Hazardous",
        "General": "🗑️ General",
        "Manual Review": "🔎 Manual Review",
        "Hidden Risk": "👜 Hidden Risk",
        "Total": "Total Objects",
        "Confidence": "Confidence (%)"
    }
)


st.dataframe(
    history_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MONITORING SUMMARY
# ============================================================

st.markdown(
    f"""
    <div class="info-box">

    <b>📡 Monitoring Status: ACTIVE</b>

    <br><br>

    The system has analyzed
    <b>{len(history_df)}</b> image(s) during this session
    and detected a total of
    <b>{total_objects}</b> waste objects.

    The dashboard tracks changes in waste-stream composition
    and AI detection confidence across successive analyses.

    </div>
    """,
    unsafe_allow_html=True
)

# ========================================================
# DEBUG INFORMATION
# ========================================================

with st.expander("🔧 Model Debug Information"):

    st.write(
        "### Classes loaded from your trained YOLO model"
    )

    st.write(
        class_names
    )

    st.write(
        f"**Objects detected:** {total}"
    )

    st.write(
        f"**Confidence threshold:** "
        f"{confidence_threshold}"
    )

    st.write(
        "The model class is shown separately from "
        "the segregation category."
    )
