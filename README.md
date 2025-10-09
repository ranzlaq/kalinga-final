# **A Mathematical Model on Breast-to-Hotspot Temperature Asymmetry and Breast Temperature Variation Through AI-Based Thermography Analysis for Breast Cancer Pre-Screening**

This repository hosts the code for KALINGA, an AI-based thermography analysis tool developed as part of a research study focusing on early breast cancer pre-screening.

The project implements a mathematical model trained under the YOLO architecture to quantify the thermal risk associated with breast tissue. The utilization concerns the insight that cancerous or pre-cancerous regions (notably termed: hotspots) exhibit significantly higher temperature asymmetry (ΔT) compared to the surrounding healthy breast tissue, and that bilateral temperature asymmetry is a key diagnostic indicator.

## **KALINGA pipeline**
+ Breast-to-Hotspot Temperature Asymmetry (ΔT): Quantifies the difference between the maximum temperature in a detected hotspot and the median temperature of the corresponding breast region.
+ Bilateral Asymmetry: Compares thermal patterns and median and standard deviation metrics between the left and right breasts.
+ Temperature Variation (σ): Analyzes the standard deviation of temperature across the breast region as a measure of thermal stability.
+ Thermal Analysis Report & Diagnostic Summary: Deliberates thermal differences and significant hotspot rise in order to provide visual evidence of detected hotspots and a comprehensive, quantitative report to aid in preliminary, non-invasive risk assessment.

## **STREAMLIT application**
KALINGA is currently released as a web-browser application. Visit it here:

[KALINGA APPLICATION](kalinga.streamlit.app)

## **GOOGLE COLAB notebook**
The final KALINGA_MODEL_FINAL.pt is a YOLOv11 model trained under ultralytics architecture in determining the following classes annotated in RoboFlow: HOTSPOT-left, HOTSPOT-right, abnormal-left, abnormal-right, normal-left, normal-right. The notebook is not publicly available; a private request must be forwarded.





