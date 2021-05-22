# Python
Python files - ML , EDA, Exercises

Please use below nbviewer link to view maps if not visible in Github ipynb notebook viewer
1. choropleth - Exercise
https://nbviewer.jupyter.org/github/mcsiva/Python/blob/main/Choropleth%20World%20Map%20-%20GDP%20per%20capita-checkpoint.ipynb

2. Classification HR Analytics - EDA, Model training & prediction
https://nbviewer.jupyter.org/github/mcsiva/Python/blob/main/Classification_HR_Analytics.ipynb

3. Credit Card Fraud Detection - 
   Training model and score done initially with Outlier detection model - Isolation Forest
   Based on scores less than 50% recall rate - Further models are built.
   XGBoost classifier -
   achieved around 99.9% accuracy with 76% recall rate for Class 1 (Fraud transacnt  class)
   RandomForest classifier -
   achieved around 99.9% accuracy with 74% recall rate for Class 1 (Fraud transacnt  class)

4. MNIST image classifier
   Following Models were trained, tuned and scored for their performance on MNIST image multiclass classification.
   RandomForest classifier, SVM Classifier(SVC), Stochastic Grad Classifier,
   TPOT Classifier 2 iterations (Log Reg clssifier and Extra trees classifier), XGBoost classifier with finetuning
  
  Model selection for MNIST dataset:
  a. Various models have been considered and performance measure have been done with accuaracy score and mean TPR
  b Based on performance metrics SVM Classifier and XGB Classifier have acc_score of (97.2%, 97.09%), mean_TPR(97.2%,97.0.9%) respectively.
  c Random Forest classifier - though accuracy is higher (99.8%), mean TPR is lower (96.1%).
  d Based on higher performance score for SVM on both accuracy and mean TPR compared to XGB --> SVM classifier performs better.
  e XGB model based on hyperparameter finetuning achieved 97% accuracy with 97.1% mean TPR. (marginal improvemnt after finetuning)
  f. SVM classifier was used to predict the result wtih final test data.

5. Product Segmentation - Model building in Azure ML cloud - Towclass BoostedTree with Onevsall Multiclass classifier and performance comparison with other models.
  https://gallery.cortanaintelligence.com/Experiment/MultiClass-label-prediction-for-product-segment-Kaggle-Dataset
