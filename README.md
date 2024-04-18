# TransForeCaster: In-and-cross categorized feature integration in user-centric deep representation learning

2-stage process: In-Category Integration (ICI) and Cross-Category Integration (CCI).

ICI employs a Time-Series Feature Mixer (TSFM) based on 1D-CNN, VAE structure.

CCI employs a Meta-Conditioned Transformer (MCT) based on Transformer architecture.

![TransForeCaster overview](https://github.com/bagelcode-data-science-team/TransForeCaster/assets/131356997/bf67100a-5d9f-4a7f-8ba4-b217962e35d5)

Our model has 690,164 parameters, 188,107,072 FLOPS.
It takes 2 hours to train the model for 3 months of data with Amazon EC2 p3.2xlarge instance.
In production, it trained weekly and inferenced daily.

## Baseline Comparison

<img width="437" alt="Baseline comparison" src="https://github.com/bagelcode-data-science-team/TransForeCaster/assets/131356997/ccbd5bd3-7979-4688-99df-a6204673bc91">

- **RFM** uses Recency, Frequency, and Monetary value of the purchase with three parametric models with different distribution assumptions on RFM: Pareto/NBD, BG/NBD, MBG/MBD. Note that RFM only utilizes recency, frequency, monetary value of the purchase and disregards the rest of the data such as user information, portrait, and behavior.
- **Two-stage XGBoost** uses a two-step process for purchase prediction. It first estimates whether a user is a payer or a non-payer, and subsequently predicts the purchase amount of the user.
- **WhalesDetector** uses a three-layer CNN (300, 150, 60 nodes with conv-pool) followed by a kernel size (7, 3, 1) to detect whether the user is a high payer (whale). We reproduced it as a regression model using ReLU at the output layer to predict LTV.
- **MSDMT** utilizes heterogeneous multi-datasource, including player portrait tabular data, behavior sequence data, and social network graph data, which leads to the comprehensive understanding of each player. Since our data does not include social network information, we employed the model excluding GNN which consists of player portraits with LSTM layers + behavior sequence with Conv-1D followed by LSTM layers and concatenated by Fully Connected layers.
- **BST** uses a transformer architecture with LeakyReLU and dropout on behavior sequence data of the user to capture interactions in sparse dataset. BST has 22,565,930 parameters, 840,821,184 FLOPS.
- **MDLUR** utilizes 3 different models for user, portrait, behavior data and concatenate it to predict purchase amount of the user.

### Parameter & FLOPs Analysis

| Models | Parameters | FLOPs | 
| ------------- | ------------- | ------------- |
| WhalesDetector  | 188,805  | 1,548,015,232  |
| MSDMT  | 372,146  | 159,686,340  |
| BST  | 22,565,930  | 840,821,184  |
| MDLUR  | 23,743,726  | 9,987,937,346  |
| **TransForeCaster(Ours)**  | **690,164**  | **188,107,072**  |
