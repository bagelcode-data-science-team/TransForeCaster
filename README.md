# TransForeCaster: In-and-Cross Categorized Feature Integration in User-Centric Deep Representation Learning

## Abstract
This paper presents **TransForeCaster**, a novel user-centric representation learning approach, designed to maximize feature integration and enhance the prediction accuracy of user purchase behavior through a two-stage process: In-Category Integration (ICI) and Cross-Category Integration (CCI). The ICI stage employs the Time-Series Feature Mixer (TSFM) for encapsulating in-category feature dynamics along the time axis, leading to compact and continuous category representations. The CCI stage then leverages a Meta-Conditioned Transformer (MCT) to integrate these representations, capturing complex cross-category relationships with metadata. This method not only facilitates precise behavior predictions but also provides interpretable insights into the factors influencing purchase behaviors. Supported by ablation studies, our methodology demonstrates significant improvements over conventional models, with in-depth evaluations using "Club Vegas Slots" game data. Additionally, we emphasize qualitative superiority through UMAP visualizations and feature importance assessments for layer-level assessments on model decisions. Extensive testing through an in-house MLOps pipeline ensures the model's robustness for live deployment. The source code is available at https://github.com/bagelcode-data-science-team/TransForeCaster

## Model Architecture Overview
### Detailed Architecture
![TransForeCaster overview](https://github.com/bagelcode-data-science-team/TransForeCaster/assets/131356997/bf67100a-5d9f-4a7f-8ba4-b217962e35d5)

### Inputs and Data Sources
- **Meta (Metadata Input)**: Includes user demographic information and other static data that provides context about the user.
- **P (Portrait Input)**: Time-series data capturing specific user interactions over time.
- **B (Behavior)**: Another set of time-series data focused on different user behaviors.

### In-Category Integration (ICI)
- **Pretraining with Time-Series Feature Mixer (TSFM)**:
  - Utilizes a Variational Autoencoder (VAE) architecture.
  - Pretrains the model on categorized features from both Portrait and Behavior data sources.
  - Captures and compresses time-series dynamics.

### Cross-Category Integration (CCI)
- **Meta-Conditioned Transformer (MCT)**:
  - Post-pretraining, utilizes a transformer architecture conditioned on metadata.
  - Integrates representations from different data categories.
  - Captures complex interrelations across different categories of data.

### Output
- **User Representation and Prediction**:
  - Combines data and learned interrelations to predict user behavior or outcomes.
  - Generates actionable insights such as user purchase behavior.

## Model Objectives
TransForeCaster aims to enhance predictive analytics by leveraging deep learning to integrate and interpret complex datasets, offering:
- **Enhanced Feature Integration**: Optimizes integration of a vast number of features.
- **Reduced Model Complexity**: Significantly reduces parameters and computational overhead.
- **Advanced Metadata Handling**: Employs Transformer-based methods for enhanced contextual understanding.

## Use Cases
- **E-commerce**: Enhances targeting and personalization strategies by predicting user purchase behavior.
- **Gaming**: Helps in customizing game experiences and increasing player retention.
- **Other Domains**: Useful wherever user behavior prediction is critical for business success.

## Comparison with [MDLUR](https://dl.acm.org/doi/pdf/10.1145/3580305.3599871) (prev. SOTA)

### Commonalities with MDLUR
- Both utilize similar datasets enriched with multiple perspectives on user data.
- Both adhere to the 1:3 undersampling ratio for class imbalance management.
- MDLUR serves as a critical benchmark for demonstrating the advancements introduced by TransForeCaster.

### Distinctions from MDLUR
- **MDLUR**: Focuses on utilizing large data spectra for varied perspectives.
- **TransForeCaster**: Maximizes feature integration across hundreds of features with a focus on model efficiency and interpretability.

### Technical Specifications
- **TransForeCaster**: 690,000 parameters, 188 million FLOPs per batch.
- **MDLUR**: 23 million parameters, 9,987 million FLOPs per batch.

## Experimental Details
- Parameters: 690,164
- FLOPs: 188,107,072
- Training Time: 2 hours for 3 months of data on Amazon EC2 p3.2xlarge instances.
- Online Settings: Batch training weekly and inferencing daily.

## Baseline Comparison

- **RFM** uses Recency, Frequency, and Monetary value of the purchase with three parametric models with different distribution assumptions on RFM: Pareto/NBD, BG/NBD, MBG/MBD. Note that RFM only utilizes recency, frequency, monetary value of the purchase and disregards the rest of the data such as user information, portrait, and behavior.
- **Two-stage XGBoost** uses a two-step process for purchase prediction. It first estimates whether a user is a payer or a non-payer, and subsequently predicts the purchase amount of the user.
- **WhalesDetector** uses a three-layer CNN (300, 150, 60 nodes with conv-pool) followed by a kernel size (7, 3, 1) to detect whether the user is a high payer (whale). We reproduced it as a regression model using ReLU at the output layer to predict LTV.
- **MSDMT** utilizes heterogeneous multi-datasource, including player portrait tabular data, behavior sequence data, and social network graph data, which leads to the comprehensive understanding of each player. Since our data does not include social network information, we employed the model excluding GNN which consists of player portraits with LSTM layers + behavior sequence with Conv-1D followed by LSTM layers and concatenated by Fully Connected layers.
- **BST** uses a transformer architecture with LeakyReLU and dropout on behavior sequence data of the user to capture interactions in sparse dataset. BST has 22,565,930 parameters, 840,821,184 FLOPS.
- **MDLUR** utilizes 3 different models for user, portrait, behavior data and concatenate it to predict purchase amount of the user.
- (Additional) 

## Baseline Performance Comparison

| Model            | MAE | RMSE | R^2 | WAP | WAR | WAF1 |
|------------------|------|------|------|------|------|------|
| RFM(Pareto/NBD)  | 62.83 | 129.80 | -0.26 | 0.0959 | 0.0398 | 0.0265 |
| RFM(BG/NBD)  | 37.52 | 93.20 | 0.35 | 0.7278 | 0.6219 | 0.6272 |
| RFM(MBG/NBD)  | 43.15 | 102.08 | 0.22 | 0.7140 | 0.5622 | 0.5729 |
| Two-stage XGBoost| 24.33 | 68.94 | -0.01 | 0.6814 | 0.7390 | 0.6975 |
| WhalesDetector   | 2.45 | 20.50 | 0.81 | 0.9651 | 0.9643 | 0.9647 |
| MSDMT            | 2.49 | 23.47 | 0.75 | 0.9705 | 0.9705 | 0.9705 |
| BST              | 2.73 | 25.99 | 0.70 | 0.9667 | 0.9665 | 0.9665 |
| MDLUR            | 2.31 | 19.54 | **0.83** | 0.9700 | 0.9695 | 0.9697 |
| **TransForeCaster** | **0.25**  | **6.99** | 0.77 | **0.9945** | **0.9948** | **0.9946** |

## Baseline Complexity Comparison

| Model            | Parameters  | FLOPs          |
|------------------|-------------|----------------|
| RFM              | N/A         | N/A            |
| Two-stage XGBoost| N/A         | N/A            |
| WhalesDetector   | 188,805     | 1,548,015,232  |
| MSDMT            | 372,146     | 159,686,340    |
| BST              | 22,565,930  | 840,821,184    |
| MDLUR            | 23,743,726  | 9,987,937,346  |
| **TransForeCaster** | **690,164**  | **188,107,072** |

**Note**: Details on RFM and Two-stage XGBoost FLOPs and parameters are not available due to its architecture.

## Conclusion
TransForeCaster showcases substantial improvements in efficiency, effectiveness, and interpretability over MDLUR, marking a significant advancement in user-centric predictive modeling.

## Getting Started
This is the simple guide to train and evaluate TransForeCaster. TransForeCaster is implemented by Tensorflow 2.0 framework. 
### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/bagelcode-data-science-team/TransForeCaster
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Running 
3. **Generate Dummy Dataset**:
    - We provide sample data, which is not real data, and code to generate the sample data.:
      ```bash
      python dummy.py
      ```

4. **Training and Evaluate**:
      ```bash
      python main.py --data ./src/data/ --lr 0.001 --batch 64 --input 7 --target 14 --epoch 15 --objective purchase
      ```
