# TransForeCaster: In-and-Cross Categorized Feature Integration in User Representation Learning

## Abstract
This paper presents **TransForeCaster**, a novel user-centric representation learning approach to improve prediction accuracy in purchase and churn predictions via a two-stage feature integration process. The In-Category Integration (ICI) stage employs a Time-Series Feature Mixer (TSFM) to capture the temporal dynamics of features within the same categories, resulting in compact and continuous category representations. The Cross-Category Integration (CCI) stage utilizes a Meta-Conditioned Transformer (MCT) to integrate these representations with multi-task learning, capturing complex relationships across categories and improving interpretability through attention mechanisms. Empirical evaluations on real-world datasets demonstrate significant improvements over conventional models, supported by qualitative analyses using feature importance assessments and UMAP visualizations. TransForeCaster's robustness is validated by its superior performance over other models in multiple in-house deployments across various games and applications.

## Model Architecture Overview
### Detailed Architecture
![TransForeCaster overview](https://github.com/user-attachments/assets/a8c7a1ca-78df-4000-9e4e-1d200c539bfb)

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
  - Generates actionable insights such as user purchase behavior and churn behavior.

## Model Objectives
TransForeCaster aims to enhance predictive analytics by leveraging deep learning to integrate and interpret complex datasets, offering:
- **Enhanced Feature Integration**: Optimizes integration of a vast number of features.
- **Reduced Model Complexity**: Significantly reduces parameters and computational overhead.
- **Advanced Metadata Handling**: Employs Transformer-based methods for enhanced contextual understanding.

## Use Cases
- **E-commerce**: Enhances targeting and personalization strategies by predicting user purchase behavior.
- **Gaming**: Helps in customizing game experiences and increasing player retention.
- **Other Domains**: Useful wherever user behavior prediction is critical for business success.

## Experimental Details
- Parameters: 1,705,471
- FLOPs: 188,107,072
- Training Time: 20 minutes for 2 months of data on Amazon EC2 p3.2xlarge instances.
- Online Settings: Batch training weekly and inferencing daily.

## Baseline Comparison

- **RFM** uses Recency, Frequency, and Monetary value of the purchase with three parametric models with different distribution assumptions on RFM: Pareto/NBD, BG/NBD, MBG/MBD. Note that RFM only utilizes recency, frequency, monetary value of the purchase and disregards the rest of the data such as user information, portrait, and behavior.
-  **Cox** is a regression model used for predicting churn among game users based on survival analysis.
- **Two-stage XGBoost** uses a two-step process for purchase prediction. It first estimates whether a user is a payer or a non-payer, and subsequently predicts the purchase amount of the user.
- **Traditional deep learning models (MLP, CNN, LSTM, Transformer)** were evaluated for fair performance comparison. A 4-layer MLP with hidden nodes configured as [128, 64, 32] was used. Additionally, a 2-layer 1D CNN and a 2-layer LSTM with hidden sizes of [32, 16] were implemented. For the Transformer model, only the encoder was used, consisting of multiple dense layers ([256, 128]) and attention mechanisms (2 layers). 
- **WhalesDetector** uses a three-layer CNN (300, 150, 60 nodes with conv-pool) followed by a kernel size (7, 3, 1) to detect whether the user is a high payer (whale). We reproduced it as a regression model using ReLU at the output layer to predict LTV.
- **MSDMT** utilizes heterogeneous multi-datasource, including player portrait tabular data, behavior sequence data, and social network graph data, which leads to the comprehensive understanding of each player. Since our data does not include social network information, we employed the model excluding GNN which consists of player portraits with LSTM layers + behavior sequence with Conv-1D followed by LSTM layers and concatenated by Fully Connected layers.
- **BST** uses a transformer architecture with LeakyReLU and dropout on behavior sequence data of the user to capture interactions in sparse dataset.
- **MDLUR** utilizes 3 different models for user, portrait, behavior data and concatenate it to predict purchase amount of the user.
- **perCLTV** utilizes heterogeneous multi-datasource, individual behavior, and social behavior. Since our data does not include social behavior information, we employed the model excluding GNN.

## Baseline Performance Comparison (Club Vegas)

| Model              | MAE      | RMSE     | R²      | F1     | AUC     |
|--------------------|----------|----------|---------|--------|---------|
| Pareto/NBD         | 47.4971  | 101.1490 | -0.2283 | -      | -       |
| BG/NBD             | 28.5572  | 57.1040  | 0.6085  | -      | -       |
| MBG/NBD            | 32.3083  | 65.9687  | 0.4775  | -      | -       |
| Cox                | -        | -        | -       | 0.8756 | 0.8480  |
| Two-stage XGBoost  | 9.2176   | 28.7999  | 0.7457  | 0.8945 | 0.8489  |
| MLP                | 1.6933   | 91.9138  | -59.1799| 0.9353 | 0.8658  |
| CNN                | 0.8647   | 38.2579  | -9.4264 | 0.9408 | 0.8748  |
| LSTM               | 0.3315   | 5.7326  | 0.7659  | 0.9412 | 0.8798  |
| Transformer        | 4.1499   | 7.0743   | 0.6435  | 0.9267 | 0.6381  |
| WhalesDetector     | 0.3702   | 5.7923  | 0.7601  | - | -  |
| MSDMT              | 0.3353   | 5.4057  | 0.7918  | 0.9406 | 0.8795  |
| BST                | 0.4508   | 6.9489  | 0.6560  | 0.9420 | 0.8836  |
| perCLTV            | 0.3754   | 6.0924   | 0.7356  | 0.9402 | 0.8775  |
| MDLUR              | 0.3121   | 7.8379   | 0.7528 | 0.9157 | 0.8796  |
| **TransForeCaster**| **0.2969** | **4.9254** | **0.8272** | **0.9422** | **0.8866** |

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

4. **Train and Evaluate**:
      ```bash
      python main.py --data ./src/data/ --lr 0.001 --batch 64 --input 7 --target 14 --epoch 15 --objective purchase
      ```
