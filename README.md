Wine Quality Prediction
This repository contains an end-to-end machine learning solution to predict the quality of red wine based on its characteristics, as described in the associated Jupyter Notebook (Wine_Quality_Prediction.ipynb). The project is based on a dataset from the UCI Machine Learning Repository and follows a structured approach to data analysis, preprocessing, model training, and evaluation.

Dataset
The dataset used in this project is the Wine Quality Dataset, which contains information about physicochemical properties (features) of red wines and their respective quality ratings.

Key Features
Predictors (Input Features): Various physicochemical properties such as acidity, sugar content, pH, and alcohol level.
Label (Output): Wine quality, rated on a scale of 0 to 10.
Project Workflow
The workflow of this project is structured as follows:

Dataset Loading:

The dataset for red wines is downloaded and loaded into the Jupyter Notebook.
Initial exploration of the dataset is performed to identify features and labels.
Exploratory Data Analysis (EDA):

Computed descriptive statistics for all features, including their ranges, types, and completeness.
Plotted histograms for each feature to understand their distributions and discussed potential improvements for features deviating from a Gaussian distribution.
Correlation Analysis:

Identified features most and least correlated with wine quality using a correlation matrix.
Discussed insights into how these correlations influence the quality prediction.
Data Preprocessing:

Split the dataset into training (80%) and testing (20%) subsets using stratified sampling to maintain statistical properties relative to quality.
Scaled the features using a StandardScaler to ensure uniform scaling for the regression model.
Model Training and Evaluation:

Trained a Linear Regression model on the scaled training data.
Evaluated model performance using:
R²-Score
Mean Absolute Error (MAE)
Mean Absolute Percentage Error (MAPE)
Mean Squared Error (MSE)
Actuals vs. Predicted plot
Commented on the results, discussing the accuracy of predictions.
Cross-Validation:

Performed 10-fold cross-validation to assess model robustness.
Calculated the mean and standard deviation of R²-scores across folds.
Compared the cross-validation results with the model's performance on the test set.
Results Summary
Descriptive Statistics: Provided insights into the feature ranges and distributions.
Feature Importance: Determined the most and least influential features on wine quality based on correlation.
Model Evaluation: Achieved satisfactory R²-scores and other performance metrics, with consistent results across 10-fold cross-validation.
Prediction Accuracy: Plotted actual vs. predicted quality scores to visualize model performance.
Repository Structure
Red_wine_quality_dataset.ipynb: Jupyter Notebook containing the full implementation of the project.
README.md: Overview and summary of the project.
Requirements
To run the Jupyter Notebook, ensure you have the following installed:

Python (3.8+)
Jupyter Notebook
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Dataset provided by the UCI Machine Learning Repository.
The assignment guidelines that inspired this project.
