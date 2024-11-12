# Wine Quality Classification RandomForest

## Overview
This project involves building a machine learning model to predict the quality of wine based on various chemical properties such as acidity, sugar content, and alcohol percentage. The model uses a Random Forest classifier to predict wine quality on a scale from 3 to 8.

## Problem Statement
The objective of this project is to accurately classify wine quality based on its chemical features. The dataset includes various features such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and the target variable, `quality`, which represents the quality of the wine.

## Approach

### 1. **Data Preprocessing**
   - **Data Exploration**: Initially, we explored the dataset to understand its structure, including the distribution of features and target values. The dataset was thoroughly examined to check for missing values and ensure data integrity.
   - **Handling Missing Values**: There were no missing values in the dataset, ensuring that the data was ready for model training.
   - **Feature Selection**: The relevant features, including both chemical properties and the target variable (quality), were retained. Features were assessed for their importance and contribution to the model’s predictions.

### 2. **Model Selection**
   - **Random Forest Classifier**: The model chosen for classification was the Random Forest classifier. It is a robust and efficient ensemble learning method that works well for classification tasks. The Random Forest model helps in handling both continuous and categorical variables and can handle large datasets with high-dimensional features.
   
### 3. **Model Tuning with Grid Search**
   - **Hyperparameter Tuning**: We used Grid Search Cross Validation (GridSearchCV) to tune the hyperparameters of the Random Forest classifier. The parameters tuned include:
     - `n_estimators`: The number of trees in the forest.
     - `max_depth`: The maximum depth of each tree.
     - `min_samples_split`: The minimum number of samples required to split an internal node.
     - `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
   - The tuning process was carried out using 5-fold cross-validation to ensure the model’s performance was optimized for generalization across different subsets of the dataset.

### 4. **Model Evaluation**
   - **Classification Report**: After tuning the model, its performance was evaluated using precision, recall, and F1-score for each class. We observed that the model performed well with an overall accuracy of approximately 80%, with precision and recall values above 0.78 for most classes.
   - **Confusion Matrix**: The confusion matrix was used to visualize the performance of the classification model, showing how well the model predicted each class and where misclassifications occurred.
   
### 5. **Final Model**
   - The best performing model, based on hyperparameter tuning, was selected as the final model. The tuned parameters provided an improved accuracy compared to the baseline model, with better classification performance across different wine quality categories.

## Results
- The model achieved an accuracy of 80% on the test set.
- The macro average precision, recall, and F1-score indicate that the model performs fairly well across all classes, though there is room for improvement, especially for the class with fewer samples (quality 2).
- The confusion matrix showed that the model struggled to classify some of the low-quality wines, likely due to the imbalance in the dataset.

## Conclusion
This project demonstrates how machine learning, specifically Random Forest, can be applied to classify wine quality based on chemical features. By tuning the model’s hyperparameters, we were able to achieve better accuracy and performance compared to the baseline model. Future improvements could include addressing the class imbalance or exploring different algorithms to further enhance predictive performance.

## Future Work
- Investigating the effect of feature scaling or transformation on model performance.
- Experimenting with other classification algorithms such as Support Vector Machines (SVM) or Gradient Boosting.
- Further analysis of class imbalance and applying techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to improve model performance for underrepresented classes.
