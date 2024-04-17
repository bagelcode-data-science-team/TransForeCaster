# TransForeCaster: In-and-cross categorized feature integration in user-centric deep representation learning

2-stage process: In-Category Integration (ICI) and Cross-Category Integration (CCI).

ICI employs a Time-Series Feature Mixer (TSFM) based on 1D-CNN, VAE structure.

CCI employs a Meta-Conditioned Transformer (MCT) based on Transformer architecture.

![TransForeCaster overview](https://github.com/bagelcode-data-science-team/TransForeCaster/assets/131356997/bf67100a-5d9f-4a7f-8ba4-b217962e35d5)

Total Time Complexity = O(epochs⋅(k⋅n⋅d^2 + n⋅d^2 + n^2⋅d + c⋅d^2))
Total Space Complexity = O(epochs⋅(k⋅d^2 + d⋅z + z⋅d + h⋅d^2 + d^2))

Our model has 690,164 parameters, 188,107,072 FLOPS.
It takes 2 hours to train the model with Amazon EC2 p3.2xlarge instance.
In production, it trained weekly and inferenced daily.

<img width="437" alt="Baseline comparison" src="https://github.com/bagelcode-data-science-team/TransForeCaster/assets/131356997/ccbd5bd3-7979-4688-99df-a6204673bc91">


Baseline
- RFM uses Recency, Frequency, and Monetary value of the purchase with three parametric models with different distribution assumptions on RFM: Pareto/NBD, BG/NBD, MBG/MBD. Note that RFM only utilizes recency, frequency, monetary value of the purchase and disregards the rest of the data such as user information, portrait, and behavior.
- Two-stage XGBoost uses a two-step process for purchase prediction. It first estimates whether a user is a payer or a non-payer, and subsequently predicts the purchase amount of the user.
- WhalesDetector uses a three-layer CNN (300, 150, 60 nodes with conv-pool) followed by a kernel size (7, 3, 1) to detect whether the user is a high payer (whale). We reproduced it as a regression model using ReLU at the output layer to predict LTV. 188,805 parameters.
- MSDMT utilizes heterogeneous multi-datasource, including player portrait tabular data, behavior sequence data, and social network graph data, which leads to the comprehensive understanding of each player. Since our data does not include social network information, we employed the model excluding GNN which consists of player portraits with LSTM layers + behavior sequence with Conv-1D followed by LSTM layers and concatenated by Fully Connected layers. 372,146 parameters.
- BST uses a transformer architecture with LeakyReLU and dropout on behavior sequence data of the user to capture interactions in sparse dataset. 22,565,930 parameters.
- MDLUR utilizes 3 different models for user, portrait, behavior data and concatenate it to predict purchase amount of the user. 23,743,726 parameters.
