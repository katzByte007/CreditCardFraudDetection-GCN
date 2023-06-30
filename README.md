# FinanceCreditCardFraudDetection-GCN
Developed an advanced GCN-based Credit Card Fraud Detection system using cutting-edge deep learning techniques. Achieved high accuracy in real-time detection and prevention of fraudulent transactions, ensuring robust security and risk mitigation in the finance industry.
Credit Card Fraud Detection using Graph Convolutional Networks (GCNs)

This project utilizes Graph Convolutional Networks (GCNs) and Deep Learning techniques to build an intelligent credit card fraud detection system. The goal is to accurately identify fraudulent transactions and provide interpretable insights for fraud analysis and risk management.

Features
Utilizes cutting-edge techniques in Graph Convolutional Networks (GCNs) for fraud detection.
Implements a deep learning model with multiple layers of GCNConv for effective feature extraction.
Applies anomaly detection methods and feature engineering to enhance fraud detection accuracy.
Incorporates explainable AI methods to provide interpretable insights into fraudulent transactions.
Dataset
The dataset used for training and evaluation is sourced from 'creditcard.csv'. It contains a large number of credit card transactions labeled as fraudulent or genuine.

Implementation
Preprocess the dataset by dropping the 'Class' column and using the remaining columns as features.
Construct a financial network graph using a fully connected graph assumption.
Train the GCN model using the training data and optimize it using the Adam optimizer.
Evaluate the model's performance using the test data and calculate the accuracy.
Iterate the training process for a specific number of epochs to improve the model's accuracy.
The final model provides accurate predictions for fraudulent transactions.
Requirements
Python 3.x
pandas
numpy
torch
torch_geometric
scikit-learn
Usage
Install the required dependencies using pip install -r requirements.txt.
Download the 'creditcard.csv' dataset and place it in the same directory as the code.
Run the 'credit_card_fraud_detection.py' script to train the GCN model and evaluate its accuracy.
Results
The accuracy achieved by the model depends on the dataset, model architecture, and training parameters. The higher the accuracy, the more effectively fraudulent transactions can be detected. The goal is to achieve a high accuracy to minimize false negatives and maximize fraud detection capabilities.

License
This project is licensed under the MIT License. Feel free to use and modify the code according to your requirements.

Acknowledgments
Special thanks to the contributors of the torch_geometric library and the authors of the original research papers on Graph Convolutional Networks for their valuable work in the field of graph-based deep learning.
