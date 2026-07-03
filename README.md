# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

This repository contains an adapted reproduction of the **ISTVT (Interpretable Spatial-Temporal Video Transformer)** model for deepfake detection, originally proposed by Zhao et al. (IEEE Transactions on Information Forensics and Security, 2023). 

This project implements the core architectural contributions of the paper—the decomposed spatial-temporal self-attention and the self-subtract mechanism—within a Google Colab notebook, optimized for a single free-tier T4 GPU.

## Overview

The ISTVT architecture addresses the limitations of frame-based deepfake detectors by leveraging both spatial and temporal inconsistencies inherent in per-frame face manipulation. The original paper's two key claims motivated this reproduction:

1.  **Decomposed Attention:** Separating spatial and temporal self-attention reduces computational complexity from $\\mathcal{O}(T^2H^2W^2)$ to $\\mathcal{O}(T^2+H^2W^2)$ while maintaining or improving performance.
2.  **Self-Subtract Mechanism:** Forcing temporal attention to focus on inter-frame residuals (rather than raw features) is critical for capturing the temporal flickering characteristic of deepfakes.

## Implementation Details & Adaptations

To operate within the constraints of a single Colab T4 GPU and a limited dataset, several necessary adaptations were made while maintaining the integrity of the core transformer architecture:

*   **Dataset:** Celeb-DF-v2 (Kaggle) was used instead of FaceForensics++ (FF++). The dataset was strictly balanced to 590 real and 590 fake videos.
*   **Feature Extractor:** The paper's from-scratch Xception backbone requires massive amounts of data (~4,000 FF++ videos) to converge. We substituted this with a **pretrained EfficientNet-B0** (timm, ImageNet weights), fine-tuned at a low learning rate. 
*   **Face Detection:** Replaced MTCNN (`facenet-pytorch`) with OpenCV's DNN face detector (SSD + ResNet-10) to resolve severe PyTorch dependency conflicts.
*   **Model Scaling:** Reduced the number of transformer blocks from 12 to 6, embedding dimension to 128, and input resolution to 128x128 to fit within the 15GB VRAM limit.
*   **Optimization:** Utilized Mixed Precision (FP16 AMP) to double throughput and layer-wise AdamW learning rates to protect the pretrained backbone while training the transformer head.

## Architecture

The implemented pipeline consists of:
1.  **Pretrained Texture Extractor:** EfficientNet-B0 producing feature maps.
2.  **ISTVT Transformer Blocks (6x):** Incorporating the self-subtract mechanism and decomposed spatial-temporal attention.
3.  **MLP Prediction Head:** Processing the joint spatial and temporal classification tokens for binary prediction.

## Results

The model was trained for 20 epochs on 944 training videos (6,608 overlapping 6-frame clips). Evaluation on the held-out validation set (236 videos) yielded:

| Metric | Value |
| :--- | :--- |
| **Clip-level Accuracy** | 93.46% |
| **Clip-level AUROC** | 0.9897 |
| **Video-level Accuracy** | 95.76% |
| **Video-level AUROC** | **0.9959** |

*Note: The confusion matrix showed a very low false-negative rate (only 2 fake videos missed out of 118).*

## Interpretability (LRP Visualization)

Algorithm 1 from the paper (Layer-wise Relevance Propagation) was faithfully implemented to generate separate spatial and temporal relevance heatmaps.

*   **Spatial Blending:** Heatmaps consistently highlight static blending boundaries (e.g., forehead/hairline).
*   **Temporal Flickering:** Heatmaps track moving facial regions (eyes, mouth) across frames, identifying inter-frame synthesis failures.

## How to Run

1.  Open the provided Google Colab notebook.
2.  Ensure you have a Kaggle API token (`kaggle.json`) ready for dataset download.
3.  Set the Colab Runtime to **T4 GPU**.
4.  Execute the notebook cells sequentially.

## Acknowledgements

This project was completed as part of the Deep Learning (EE656) course at the Indian Institute of Technology Kanpur (IITK), instructed by Prof. Nishchal K. Verma.

**Authors:** Ojas Bajpai, Pratyush Juyal, Priyanshu Yadav, Snehansh Maharaj, Gourav Singh Chouhan.
