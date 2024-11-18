# Artificial Neural Networks (ANN) Script using h2o
This R script demonstrates the implementation of an Artificial Neural Network (ANN) for predicting customer churn based on a dataset. It leverages the h2o package for building and training the ANN model, enabling efficient and scalable computations.

## Key Steps in the Script
### **1. Importing the Dataset**
- The dataset (Churn_Modelling.csv) is imported and columns irrelevant to the analysis are excluded, starting from the 4th column to the last (4:ncol(dataset)).
  
### **2. Encoding Categorical Variables**
- Geography: Encoded into numeric values corresponding to France (1), Spain (2), and Germany (3).
- Gender: Converted to binary values 0 (No) and 1 (Yes).
  
### **3. Splitting the Data**
- The dataset is divided into training (80%) and test (20%) sets using the caTools library.
- The Exited column is used as the target variable to stratify the split.
  
### **4. Feature Scaling**
All independent variables are scaled (standardized) for better model performance, excluding the target column (Exited).

### **5. Building and Training the ANN**
- The h2o.deeplearning function is used to construct and train the ANN:
- Target Variable: Exited.
- Activation Function: Rectifier (ReLU) for non-linearity.
- Hidden Layers: Two layers with 6 neurons each.
- Epochs: 100 iterations through the training set.
- Batch Size: Automatically determined using train_samples_per_iteration = -2.
  
### **6. Predicting Test Set Results**
- The model predicts probabilities for the test set, which are thresholded at 0.5 to classify observations as churned (1) or not churned (0).
  
### **7. Evaluating the Model**
- A confusion matrix is created to evaluate model performance.
- Example of metrics:
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)

8. Accuracy Calculation
Model accuracy is calculated as:
Accuracy
=
TP
+
TN
Total Observations
Accuracy= 
Total Observations
TP+TN
​
 
The script achieves an accuracy of 85.85% (0.8585).
9. Clean-Up
The H2O session is shut down at the end to release resources.
Why Use H2O for ANN?
High Performance: Utilizes all available CPU cores for parallel computation.
Scalability: Handles large datasets effectively.
Ease of Use: Provides a high-level API for creating complex neural networks with minimal configuration.
Possible Enhancements
Hyperparameter Tuning: Experiment with different numbers of neurons, layers, activation functions, and learning rates.
Feature Selection: Perform feature importance analysis to remove irrelevant variables.
Cross-Validation: Use k-fold cross-validation to ensure the model generalizes well.
Performance Metrics: Evaluate additional metrics like precision, recall, F1-score, and ROC-AUC for deeper insights.
