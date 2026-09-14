"#DefectDetection"
# AI Visual Defect Detection for Manufacturing

## Project Overview

AI Visual Defect Detection is an automated visual inspection system for manufacturing quality control.

The system analyzes product images and predicts whether the product is:

- ✅ NOT DEFECTIVE / NORMAL
- ❌ DEFECTIVE

It also generates an anomaly heatmap to highlight the suspected defect region.

## Objective

To develop an AI-based system that can automatically identify manufacturing defects from product images using unsupervised anomaly detection.

## Dataset

The project uses the **MVTec AD (Anomaly Detection) dataset**.

Initial supported product categories:

- Screw
- Bottle
- Metal Nut

Each category contains normal training images and normal/defective testing images.

## Method Used

### PatchCore

PatchCore is used as the main anomaly detection model.

The model is trained using only normal (`train/good`) images. During testing, it compares the input image with learned normal patterns and produces an anomaly score.

## System Workflow

```text
Input Image
     ↓
Product Category
     ↓
PatchCore Model
     ↓
Anomaly Score
     ↓
NORMAL / DEFECTIVE
     ↓
Anomaly Heatmap
