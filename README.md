# Real-Time Risk Assessment Framework

This repository contains the implementation of a real-time risk assessment framework for vehicular security incidents. The framework consists of two main phases: **Offline Phase** and **Online Phase**, implemented in `main.py`. Additionally, it includes supporting modules for classification, clustering, and incident data processing.

## Repository Structure

├── Code/ │ ├── classify.py # Multi-label classification of security incidents │ ├── cluster.py # Clustering-based incident categorization │ ├── main.py # Implementation of offline and online phases │ ├── read_incidents.py # Data preprocessing and feature encoding │ ├── Files/ │ ├── dataframe2.csv # Mapped dataset (confidential information removed)

## Implementation Details

### Offline Phase (`main.py`)
The offline phase involves preprocessing security incident data, training machine learning models, and evaluating their performance.

- **Data Loading & Preparation:**  
  - Reads the dataset (`dataframe2.csv`) and applies feature encoding.
  - Splits data into **training/testing (80%)** and **real-time evaluation (10%)** sets.

- **Multi-label Classification:**  
  - Prepares data for classification.
  - Implements a two-layer classification model to predict security incident attributes.
  - Evaluates and visualizes classification performance.

- **Clustering:**  
  - Finds optimal clusters for security incidents.
  - Assigns each incident to a cluster for further analysis.

### Online Phase (`main.py`)
The online phase performs real-time classification and clustering of new security incidents while continuously adapting the models.

- **Feature Extraction & Preprocessing:**  
  - Aligns incoming data with expected model features.
  - Handles missing values and categorical encoding.

- **Real-Time Clustering & Classification:**  
  - Predicts the cluster of new incidents.
  - Updates the clustering model dynamically.
  - Performs classification and updates models periodically.

- **Decision-Making & Risk Assessment:**  
  - Computes criticality based on classification and clustering results.
  - Weighs security impact factors and determines risk severity.
  - Stores and visualizes real-time evaluation results.

## Confidentiality Notice
The dataset in the `Files/` directory has been modified to remove information about vehicular architecture due to confidentiality reasons. However, the mapped dataset retains its structural integrity for research and implementation purposes.

## Usage
To run the implementation, execute:

```bash
python main.py
