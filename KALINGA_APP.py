import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.patches import Rectangle
import textwrap
import matplotlib.gridspec as gridspec
from PIL import Image
import io

# STREAMLIT STARTUP
st.set_page_config(
    page_title="KALINGA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CONSTANTS ---
MODEL_PATH = "KALINGA_MODEL_FINAL.pt"
MIN_TEMP = 32.0
MAX_TEMP = 41.0
HOTSPOT_PCT = 95.0
MIN_DELTA_FOR_HOTSPOT = 0.2 
MAX_DT_THRESHOLD = 3.0

# --- UTILITY FUNCTIONS ---
@st.cache_resource
def load_model(path):
    """Load the YOLO model, cached for efficiency."""
    try:
        model = YOLO(path) 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def pixel_to_temp(pixel_val, min_temp=MIN_TEMP, max_temp=MAX_TEMP):
    """Converts a pixel value (0-255) to a temperature (°C)."""
    return min_temp + (pixel_val / 255.0) * (max_temp - min_temp)

# -------------------- ASYMMETRY ANALYSIS --------------------
def calculate_asymmetry(image_rgb, left_box, right_box,
                        hotspot_pct=HOTSPOT_PCT,
                        min_delta_for_hotspot=MIN_DELTA_FOR_HOTSPOT,
                        resize_for_diffmap=(100,100)):

    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = image_gray.shape[:2]

    l_x1, l_y1, l_x2, l_y2 = map(int, left_box)
    r_x1, r_y1, r_x2, r_y2 = map(int, right_box)

    l_x1 = max(0, min(w-1, l_x1)); l_x2 = max(0, min(w, l_x2))
    r_x1 = max(0, min(w-1, r_x1)); r_x2 = max(0, min(w, r_x2))
    l_y1 = max(0, min(h-1, l_y1)); l_y2 = max(0, min(h, l_y2))
    r_y1 = max(0, min(h-1, r_y1)); r_y2 = max(0, min(h, r_y2))

    if l_x2 <= l_x1 or l_y2 <= l_y1 or r_x2 <= r_x1 or r_y2 <= r_y1:
        st.error("Invalid bbox coordinates after boundary check.")
        return None

    left_roi_gray = image_gray[l_y1:l_y2, l_x1:l_x2]
    right_roi_gray = image_gray[r_y1:r_y2, r_x1:r_x2]

    if left_roi_gray.size == 0 or right_roi_gray.size == 0:
        st.error("Empty left/right crop - check bounding boxes")
        return None

    mean_left_px = float(np.mean(left_roi_gray))
    mean_right_px = float(np.mean(right_roi_gray))
    median_left_px = float(np.median(left_roi_gray))
    median_right_px = float(np.median(right_roi_gray))

    mean_left = pixel_to_temp(mean_left_px)
    mean_right = pixel_to_temp(mean_right_px)
    median_left = pixel_to_temp(median_left_px)
    median_right = pixel_to_temp(median_right_px)

    hotspot_thresh_left_px = np.percentile(left_roi_gray, hotspot_pct)
    hotspot_thresh_right_px = np.percentile(right_roi_gray, hotspot_pct)

    left_hotspots_px = left_roi_gray[left_roi_gray >= hotspot_thresh_left_px]
    right_hotspots_px = right_roi_gray[right_roi_gray >= hotspot_thresh_right_px]

    # MAX PIXEL VALUES IN HOTSPOT AREA
    left_hotspot_max_px = float(np.max(left_hotspots_px)) if left_hotspots_px.size > 0 else 0.0
    right_hotspot_max_px = float(np.max(right_hotspots_px)) if right_hotspots_px.size > 0 else 0.0
    left_hotspot_mean_px = float(np.mean(left_hotspots_px)) if left_hotspots_px.size > 0 else 0.0
    right_hotspot_mean_px = float(np.mean(right_hotspots_px)) if right_hotspots_px.size > 0 else 0.0

    # TEMPERATURE CONVERSION
    left_hotspot_max_temp = pixel_to_temp(left_hotspot_max_px)
    right_hotspot_max_temp = pixel_to_temp(right_hotspot_max_px)
    left_hotspot_mean_temp = pixel_to_temp(left_hotspot_mean_px)
    right_hotspot_mean_temp = pixel_to_temp(right_hotspot_mean_px)

    # DELTA T CALCULATIONS
    delta_left_max = left_hotspot_max_temp - median_left
    delta_right_max = right_hotspot_max_temp - median_right
    delta_left_mean = left_hotspot_mean_temp - median_left
    delta_right_mean = right_hotspot_mean_temp - median_right

    left_hotspot_size = int(left_hotspots_px.size) if (left_hotspots_px.size > 0 and delta_left_max > min_delta_for_hotspot) else 0
    right_hotspot_size = int(right_hotspots_px.size) if (right_hotspots_px.size > 0 and delta_right_max > min_delta_for_hotspot) else 0

    left_gray_temp = pixel_to_temp(left_roi_gray)
    right_gray_temp = pixel_to_temp(right_roi_gray)
    std_left = np.std(left_gray_temp)
    std_right = np.std(right_gray_temp)

    hist_left = cv2.calcHist([left_roi_gray.astype(np.uint8)], [0], None, [256], [0,256])
    hist_right = cv2.calcHist([right_roi_gray.astype(np.uint8)], [0], None, [256], [0,256])
    cv2.normalize(hist_left, hist_left, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_right, hist_right, 0, 1, cv2.NORM_MINMAX)
    histogram_diff = float(cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_BHATTACHARYYA))

    lr = cv2.resize(left_roi_gray, resize_for_diffmap, interpolation=cv2.INTER_LINEAR)
    rr = cv2.resize(right_roi_gray, resize_for_diffmap, interpolation=cv2.INTER_LINEAR)
    temp_diff_map = cv2.absdiff(lr.astype(np.float32), rr.astype(np.float32))
    significant_diff_ratio = (np.sum(temp_diff_map > (0.15 * 255.0)) / float(temp_diff_map.size)) if temp_diff_map.size > 0 else 0.0

    return {
        "delta_left_mean": delta_left_mean, "delta_left_max": delta_left_max,
        "delta_right_mean": delta_right_mean, "delta_right_max": delta_right_max,
        "left_hotspot_max_temp": left_hotspot_max_temp,
        "right_hotspot_max_temp": right_hotspot_max_temp,
        "left_hotspot_size": left_hotspot_size, "right_hotspot_size": right_hotspot_size,
        "temp_diff_map": temp_diff_map,
        "mean_left": mean_left, "mean_right": mean_right,
        "median_left": median_left, "median_right": median_right,
        "std_left": std_left, "std_right": std_right,
        "histogram_diff": histogram_diff,
        "significant_diff_ratio": significant_diff_ratio
    }

# -------------------- TEXT SUMMARY --------------------
def generate_text_explanation(
    mean_left, mean_right, significant_diff_ratio,
    delta_left_max, delta_right_max,
    hotspot_any_detected
):
    
    if not hotspot_any_detected:
        return (
            "Thermal analysis shows no detectable hotspot temperature rise. "
            "Both breasts appear normal based on the absence of hotspots. "
            "However, regular screening is strongly advised, especially if you experience any symptoms. "
            "Consider repeating the thermogram analysis for continued monitoring."
        )
    
    dt_left_report = max(0, delta_left_max)
    dt_right_report = max(0, delta_right_max)

    if significant_diff_ratio > 0.50:
        distribution_insight = f"Thermal differences are widespread across {significant_diff_ratio*100:.1f}% of the tissue. "
    elif significant_diff_ratio > 0.30:
        distribution_insight = f"Thermal differences affect {significant_diff_ratio*100:.1f}% of the tissue. "
    else:
        distribution_insight = f"Thermal differences are isolated to small regions ({significant_diff_ratio*100:.1f}%). "

    if dt_left_report > 0.5 or dt_right_report > 0.5:
        hotspot_insight = (
            f"Significant hotspot temperature rise detected "
            f"(ΔT Left: {dt_left_report:.2f}°C, ΔT Right: {dt_right_report:.2f}°C). "
        )
    elif dt_left_report > 0.2 or dt_right_report > 0.2:
        hotspot_insight = (
            f"Moderate hotspot temperature rise detected "
            f"(ΔT Left: {dt_left_report:.2f}°C, ΔT Right: {dt_right_report:.2f}°C). "
        )
    else:
        hotspot_insight = (
            f"Minimal hotspot temperature rise detected "
            f"(ΔT Left: {dt_left_report:.2f}°C, ΔT Right: {dt_right_report:.2f}°C). "
        )

    technical_details = distribution_insight + hotspot_insight

    return (
        "Hotspot regions detected during analysis. "
        f"{technical_details}Immediate consultation with a medical professional is strongly advised."
    )


def get_interpretation_text(side, hotspot_detected, hotspot_count, delta_max, std):
    """Generates the detailed per-breast interpretation text from the original script."""
    
    if not hotspot_detected or delta_max <= MIN_DELTA_FOR_HOTSPOT:
        return (
            f"In the {side} breast region, thermal analysis shows no detectable anomaly "
            f"or hotspot temperature rise but pattern variation of $\pm{std:.2f}^\circ C$ is observed."
        )

    if hotspot_count > 1:
        if delta_max > 2:
            detail = "significant temperature difference"
        elif delta_max > 1:
            detail = "moderate temperature difference"
        else:
            detail = "minimal temperature difference"
        
        return (
            f"In the {side} breast region, {detail} of {delta_max:.2f}$^\circ C$ "
            f"is detected between breast and hottest hotspot. Multifocal hotspots are also detected, and there is a possibility that different types of anomalies are present."
        )
    else:
        if delta_max > 2:
            detail = "significant temperature difference"
        elif delta_max > 1:
            detail = "moderate temperature difference"
        else:
            detail = "minimal temperature difference"
            
        return (
            f"In the {side} breast region, {detail} of {delta_max:.2f}$^\circ C$ "
            f"is detected between breast and hottest hotspot. Only one significant hotspot is detected."
        )

# -------------------- PLOTTING UTILITIES --------------------
def draw_asymmetry_bar(ax, side_label, delta, hotspot_detected=True, MAX_DT=MAX_DT_THRESHOLD):
    """
    Draws a temperature delta bar and calculates the breast-specific percentage,
    including colored sections and handling for 'no hotspot detected'.
    """
    bar_x = 0.1
    bar_width, bar_height = 0.8, 0.1
    label_offset = 0.02
    bar_y = 0.75

    ax.add_patch(Rectangle((bar_x, bar_y), bar_width, bar_height, color="lightgray"))

    if hotspot_detected and delta > MIN_DELTA_FOR_HOTSPOT: 
        delta_clamped = max(0.0, float(delta))
        score_clamped = min(delta_clamped / MAX_DT, 1.0)
        percentage = score_clamped * 100.0
        
        # COLOUR THRESHOLDS
        if score_clamped < 0.33:
            fill_color = "green"
        elif score_clamped < 0.66:
            fill_color = "orange"
        else:
            fill_color = "red"

        ax.add_patch(Rectangle((bar_x, bar_y), bar_width * score_clamped, bar_height, color=fill_color))
        ax.text(0.5, bar_y + bar_height + label_offset,
                f"{side_label} Breast Asymmetry: $\Delta T$={delta:.2f}$^\circ C$ ({percentage:.1f}%)", 
                ha="center", va="bottom", fontsize=14)
    else:
        ax.text(0.5, bar_y + bar_height + label_offset,
                f"{side_label} Breast Asymmetry: 0% (No Hotspot Detected)", 
                ha="center", va="bottom", fontsize=14, color="gray")

    ax.text(bar_x, bar_y - 0.05, "0$^\circ C$ - Low", ha="left", va="top", fontsize=10, color="green")
    ax.text(bar_x + bar_width, bar_y - 0.05, f"{MAX_DT}$^\circ C$ - High", ha="right", va="top", fontsize=10, color="red")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

def find_hottest_hotspot_detection(detections, side_tag, image_gray):
    """
    Finds the specific hotspot detection box with the highest *internal* max temperature 
    for the given side.
    """
    hottest_det = None
    max_temp_val = -float('inf')
    
    hotspot_detections = [d for d in detections if side_tag in d["class_name"].lower()]

    for det in hotspot_detections:
        x1, y1, x2, y2 = map(int, det["box"])
        hotspot_region = image_gray[y1:y2, x1:x2]
        if hotspot_region.size > 0:
            max_val_px = np.max(hotspot_region)
            max_temp_cv2 = pixel_to_temp(max_val_px)
            
            if max_temp_cv2 > max_temp_val:
                max_temp_val = max_temp_cv2
                hottest_det = det
                
    return hottest_det, max_temp_val

def create_report_figure(rgb_img, detections, stats):

    delta_left_max = max(0, stats.get("delta_left_max", 0.0))
    delta_right_max = max(0, stats.get("delta_right_max", 0.0))
    
    is_thermally_hotspot_left = delta_left_max > MIN_DELTA_FOR_HOTSPOT
    is_thermally_hotspot_right = delta_right_max > MIN_DELTA_FOR_HOTSPOT
    
    left_hotspot_count = sum(1 for d in detections if "hotspot-left" in d["class_name"].lower())
    right_hotspot_count = sum(1 for d in detections if "hotspot-right" in d["class_name"].lower())
    
    # BOX OVERRIDE
    hotspot_detected_by_yolo_left = left_hotspot_count > 0
    hotspot_detected_by_yolo_right = right_hotspot_count > 0

    # HOTSPOTS MUST BYPASS THERMAL THRESHOLD AND YOLO DETECTIONS
    hotspot_left_detected = is_thermally_hotspot_left and hotspot_detected_by_yolo_left
    hotspot_right_detected = is_thermally_hotspot_right and hotspot_detected_by_yolo_right
    hotspot_any_detected = hotspot_left_detected or hotspot_right_detected

    median_left = stats.get("median_left", 0.0)
    median_right = stats.get("median_right", 0.0)
    mean_left = stats.get("mean_left", 0.0)
    mean_right = stats.get("mean_right", 0.0)
    std_left = stats.get("std_left", 0.0)
    std_right = stats.get("std_right", 0.0)
    significant_diff_ratio = stats.get("significant_diff_ratio", 0.0)

    left_hotspot_max_temp = stats.get("left_hotspot_max_temp", 0.0)
    right_hotspot_max_temp = stats.get("right_hotspot_max_temp", 0.0)
    
    display_delta_left = delta_left_max if hotspot_left_detected else 0.00
    display_delta_right = delta_right_max if hotspot_right_detected else 0.00
    display_max_temp_left = left_hotspot_max_temp if hotspot_left_detected else 0.00
    display_max_temp_right = right_hotspot_max_temp if hotspot_right_detected else 0.00

    text_summary = generate_text_explanation(
        mean_left, mean_right, significant_diff_ratio,
        delta_left_max, delta_right_max,
        hotspot_any_detected
    )
    interpretation_left = get_interpretation_text("left", hotspot_left_detected, left_hotspot_count, delta_left_max, std_left)
    interpretation_right = get_interpretation_text("right", hotspot_right_detected, right_hotspot_count, delta_right_max, std_right)

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    
    # -------------------- PLOTTING WITH 3-ROW GRID --------------------
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.5, 0.75, 1], width_ratios=[1, 1])
    plt.subplots_adjust(hspace=0.25, top=0.95, bottom=0.05)

    # === YOLO DETECTIONS ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_img)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_name = det["class_name"]
        conf = det["conf"]
        color = "red" if "hotspot" in class_name.lower() and hotspot_any_detected else "green"
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 5, f"{class_name}: {conf:.2f}", color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.7))
    ax1.set_title("YOLOv11 Mirrored Breast and Hotspot Detections", fontsize=16)
    ax1.axis("off")

    # === MAXIMUM HOTSPOT TEMPERATURE ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gray_img, cmap="gray")
    ax2.set_title("Maximum Hotspot Temperature", fontsize=16)
    ax2.axis("off")

    # --- LEFT SIDE Hottest Hotspot ---
    if hotspot_left_detected:
        hottest_left_det, max_temp_cv2_left = find_hottest_hotspot_detection(detections, "hotspot-left", gray_img)
        if hottest_left_det:
            x1, y1, x2, y2 = map(int, hottest_left_det["box"])
            
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
            ax2.add_patch(rect)
            
            hotspot_region = gray_img[y1:y2, x1:x2]
            max_loc = np.unravel_index(np.argmax(hotspot_region), hotspot_region.shape)
            hot_y, hot_x = max_loc
            hot_x += x1
            hot_y += y1
            ax2.plot(hot_x, hot_y, "ro", markersize=6)
            
            ax2.text(
                x1, y1 - 15,
                f"CV2 Luminosity: {max_temp_cv2_left:.2f}°C\nROI Percentile: {left_hotspot_max_temp:.2f}°C",
                color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.7)
            )

    # --- RIGHT SIDE Hottest Hotspot ---
    if hotspot_right_detected:
        hottest_right_det, max_temp_cv2_right = find_hottest_hotspot_detection(detections, "hotspot-right", gray_img)
        if hottest_right_det:
            x1, y1, x2, y2 = map(int, hottest_right_det["box"])

            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
            ax2.add_patch(rect)
            
            hotspot_region = gray_img[y1:y2, x1:x2]
            max_loc = np.unravel_index(np.argmax(hotspot_region), hotspot_region.shape)
            hot_y, hot_x = max_loc
            hot_x += x1
            hot_y += y1
            ax2.plot(hot_x, hot_y, "ro", markersize=6)

            ax2.text(
                x1, y1 - 15,
                f"CV2 Luminosity: {max_temp_cv2_right:.2f}°C\nROI Percentile: {right_hotspot_max_temp:.2f}°C",
                color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.7)
            )


    # === LEFT METRICS ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Left Breast Metrics", fontsize=16, fontweight='bold', pad=4)
    draw_asymmetry_bar(ax3, "Left", delta_left_max, hotspot_detected=hotspot_left_detected)

    y_start, line_spacing = 0.50, 0.08 
    
    # 1. Max Hotspot Temp
    ax3.text(0.25, y_start, "Max Hotspot Temp:", fontsize=12, va="center", ha="left")
    ax3.text(0.75, y_start, f"{display_max_temp_left:.2f}$^\circ C$", fontsize=12, va="center", ha="right")
    
    # 2. Median Breast Temp
    ax3.text(0.25, y_start - line_spacing, "Median Breast Temp:", fontsize=12, va="center", ha="left")
    ax3.text(0.75, y_start - line_spacing, f"{median_left:.2f}$^\circ C$", fontsize=12, va="center", ha="right")
    
    # 3. Max Hotspot - Median Breast Temp
    ax3.text(0.25, y_start - 2*line_spacing, "Max Hotspot - Median Breast Temp:", fontsize=12, va="center", ha="left")
    ax3.text(0.75, y_start - 2*line_spacing, f"{display_delta_left:.2f}$^\circ C$", fontsize=12, va="center", ha="right")

    # 4. Mean Breast Temp
    ax3.text(0.25, y_start - 3*line_spacing, "Mean Breast Temp:", fontsize=12, va="center", ha="left")
    ax3.text(0.75, y_start - 3*line_spacing, f"{mean_left:.2f}$^\circ C$", fontsize=12, va="center", ha="right")

    # 5. No. of Hotspots Detected / Temperature Variation (CONDITIONAL)
    if hotspot_left_detected:
        ax3.text(0.25, y_start - 4*line_spacing, "No. of Hotspots Detected:", fontsize=12, va="center", ha="left")
        ax3.text(0.75, y_start - 4*line_spacing, f"{left_hotspot_count}", fontsize=12, va="center", ha="right")
    else:
        ax3.text(0.25, y_start - 4*line_spacing, "Temperature Variation:", fontsize=12, va="center", ha="left")
        ax3.text(0.75, y_start - 4*line_spacing, f"$\pm{std_left:.2f}^\circ C$", fontsize=12, va="center", ha="right")

    wrapped_left = textwrap.fill(interpretation_left, width=60)
    ax3.text(0.5, 0.05, wrapped_left, fontsize=12, va="top", ha="center")
    ax3.axis("off")

    # === RIGHT METRICS ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Right Breast Metrics", fontsize=16, fontweight='bold', pad=4)
    draw_asymmetry_bar(ax4, "Right", delta_right_max, hotspot_detected=hotspot_right_detected)

    y_start, line_spacing = 0.50, 0.08
    
    # 1. Max Hotspot Temp
    ax4.text(0.25, y_start, "Max Hotspot Temp:", fontsize=12, va="center", ha="left")
    ax4.text(0.75, y_start, f"{display_max_temp_right:.2f}$^\circ C$", fontsize=12, va="center", ha="right")
    
    # 2. Median Breast Temp
    ax4.text(0.25, y_start - line_spacing, "Median Breast Temp:", fontsize=12, va="center", ha="left")
    ax4.text(0.75, y_start - line_spacing, f"{median_right:.2f}$^\circ C$", fontsize=12, va="center", ha="right")
    
    # 3. Max Hotspot - Median Breast Temp
    ax4.text(0.25, y_start - 2*line_spacing, "Max Hotspot - Median Breast Temp:", fontsize=12, va="center", ha="left")
    ax4.text(0.75, y_start - 2*line_spacing, f"{display_delta_right:.2f}$^\circ C$", fontsize=12, va="center", ha="right")

    # 4. Mean Breast Temp
    ax4.text(0.25, y_start - 3*line_spacing, "Mean Breast Temp:", fontsize=12, va="center", ha="left")
    ax4.text(0.75, y_start - 3*line_spacing, f"{mean_right:.2f}$^\circ C$", fontsize=12, va="center", ha="right")

    # 5. No. of Hotspots Detected / Temperature Variation (CONDITIONAL)
    if hotspot_right_detected:
        ax4.text(0.25, y_start - 4*line_spacing, "No. of Hotspots Detected:", fontsize=12, va="center", ha="left")
        ax4.text(0.75, y_start - 4*line_spacing, f"{right_hotspot_count}", fontsize=12, va="center", ha="right")
    else:
        ax4.text(0.25, y_start - 4*line_spacing, "Temperature Variation:", fontsize=12, va="center", ha="left")
        ax4.text(0.75, y_start - 4*line_spacing, f"$\pm{std_right:.2f}^\circ C$", fontsize=12, va="center", ha="right")

    wrapped_right = textwrap.fill(interpretation_right, width=60)
    ax4.text(0.5, 0.05, wrapped_right, fontsize=12, va="top", ha="center")
    ax4.axis("off")

    # === DIAGNOSTIC SUMMARY ===
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis("off")
    ax_summary.text(0.5, 0.85, "Diagnostic Summary", fontsize=25, fontweight='bold', ha="center")
    wrapped_summary = textwrap.fill(text_summary, width=120)
    ax_summary.text(0.5, 0.4, wrapped_summary, fontsize=16, va="center", ha="center")

    plt.tight_layout()
    return fig

# -------------------- STREAMLIT MAIN APPLICATION --------------------
def main():
    st.title("A Mathematical Model on Breast-to-Hotspot Temperature Asymmetry and Breast Temperature Variation Through AI-Based Thermography Analysis for Breast Cancer Pre-Screening")
    st.markdown("")
    st.markdown("[DESCRIPTION TOLLLL]")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a thermogram (blue-red palette)...", type=["jpg", "jpeg", "png"])

    model = load_model(MODEL_PATH)
    if model is None:
        st.stop() 

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode image.")
            return

        image = cv2.flip(image, 1)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with st.spinner("Running YOLOv11 detection and thermal analysis..."):
            results = model(image)

            left_breast_box, right_breast_box = None, None
            detections_list = []
            
            for result in results:
                for det in result.boxes.data:
                    x1, y1, x2, y2, conf, class_idx = det.cpu().numpy()
                    class_name = model.names[int(class_idx)]
                    box = [x1, y1, x2, y2]

                    if "left" in class_name.lower() and "hotspot" not in class_name.lower():
                        left_breast_box = box
                    elif "right" in class_name.lower() and "hotspot" not in class_name.lower():
                        right_breast_box = box

                    detections_list.append({"box": box, "conf": conf, "class_name": class_name})

            if left_breast_box is None or right_breast_box is None:
                st.error("Error: Could not detect both left and right breast regions in the image.")
                return
            else:
                try:
                    stats = calculate_asymmetry(rgb_img, left_breast_box, right_breast_box)

                    if stats is None:
                        return 
                    
                    delta_left_max = max(0, stats.get("delta_left_max", 0.0))
                    delta_right_max = max(0, stats.get("delta_right_max", 0.0))
                    
                    left_hotspot_count = sum(1 for d in detections_list if "hotspot-left" in d["class_name"].lower())
                    right_hotspot_count = sum(1 for d in detections_list if "hotspot-right" in d["class_name"].lower())
                    
                    hotspot_left_detected = (delta_left_max > MIN_DELTA_FOR_HOTSPOT) and (left_hotspot_count > 0)
                    hotspot_right_detected = (delta_right_max > MIN_DELTA_FOR_HOTSPOT) and (right_hotspot_count > 0)

                    fig = create_report_figure(rgb_img, detections_list, stats)
                    st.pyplot(fig)

                    original_name = uploaded_file.name.rsplit('.', 1)[0]
                    custom_filename = f"thermal_analysis_report_{original_name}.png"

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col2:
                        st.download_button(
                            label="Download Full Analysis Report",
                            data=buf.getvalue(),
                            file_name=custom_filename,
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    st.markdown("---")
                    
                    st.markdown("<h2 style='text-align: center;'>Raw Asymmetry Metrics</h2>", unsafe_allow_html=True)
                    st.markdown("")
                    
                    max_temp_L = stats["left_hotspot_max_temp"] if hotspot_left_detected else 0.00
                    max_temp_R = stats["right_hotspot_max_temp"] if hotspot_right_detected else 0.00
                    delta_max_L = max(0, stats["delta_left_max"]) if hotspot_left_detected else 0.00
                    delta_max_R = max(0, stats["delta_right_max"]) if hotspot_right_detected else 0.00
                    delta_mean_L = max(0, stats["delta_left_mean"]) if hotspot_left_detected else 0.00
                    delta_mean_R = max(0, stats["delta_right_mean"]) if hotspot_right_detected else 0.00

                    # RAW METRICS TABLE
                    st.markdown(f"""
                        | Metric | Left Breast | Right Breast | Bilateral |
                        | :--- | :--- | :--- | :--- |
                        | **Median Temp** | {stats["median_left"]:.2f} $^\circ C$ | {stats["median_right"]:.2f} $^\circ C$ | **Diff**: {abs(stats["median_left"] - stats["median_right"]):.2f} $^\circ C$ |
                        | **Mean Temp** | {stats["mean_left"]:.2f} $^\circ C$ | {stats["mean_right"]:.2f} $^\circ C$ | **Diff**: {abs(stats["mean_left"] - stats["mean_right"]):.2f} $^\circ C$ |
                        | **Max Hotspot Temp** | {max_temp_L:.2f} $^\circ C$ | {max_temp_R:.2f} $^\circ C$ | N/A |
                        | **Max $\Delta T$ (Hotspot-Median)** | {delta_max_L:.2f} $^\circ C$ | {delta_max_R:.2f} $^\circ C$ | N/A |
                        | **Mean $\Delta T$ (Hotspot-Median)** | {delta_mean_L:.2f} $^\circ C$ | {delta_mean_R:.2f} $^\circ C$ | N/A |
                        | **Temp Variation (Std Dev)** | $\pm{stats["std_left"]:.2f}$ $^\circ C$ | $\pm{stats["std_right"]:.2f}$ $^\circ C$ | N/A |
                        | **Hotspot Area (Pixels)** | {stats["left_hotspot_size"]} | {stats["right_hotspot_size"]} | N/A |
                        | **Histogram Difference (Bhattacharyya)** | N/A | N/A | {stats["histogram_diff"]:.4f} |
                        | **Significant Diff Ratio (>0.15 $\times$ 255)** | N/A | N/A | {stats["significant_diff_ratio"]*100:.1f} % |
                    """)

                except Exception as e:
                    st.error(f"An error occurred during thermal analysis: {e}")
                    st.exception(e)

if __name__ == "__main__":

    main()
