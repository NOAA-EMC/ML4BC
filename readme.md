# GFS Bias Correction using ML

This project implements a deep learning model in PyTorch for bias correction of GFS (Global Forecast System) temperature forecasts using ERA5 data (ground truth) as a reference. The goal is to reduce forecast biases in GFS 2m temperature forecasts by training a neural network to predict unbiased GFS data based on ERA5 observations.

## Table of Contents

- [Background](#background)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Bias Correction](#bias-correction)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## Background

The GFS is a widely used weather forecasting system, but it may contain biases in temperature predictions. Bias correction is essential to improve the accuracy of these forecasts. This project presents a PyTorch-based bias correction model that learns to predict unbiased GFS temperature forecasts by comparing them with ERA5 data, which serves as the ground truth.

## Getting Started

### Prerequisites

Before using this project, make sure you have a workspace with the following libraries installed:

- Python 3.x
- PyTorch
- NumPy
- Xarray
- Other required libraries (e.g., Matplotlib, Pandas)

