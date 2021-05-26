# Python
Python files - ML , EDA, Exercises

1. **Choropleth - Folium maps module**
   Exercise on Folium maps module to view maps, *import GeoJSON file and apply categorical metrics in maps*.
   Please use below *nbviewer* link to view maps if not visible in *Github ipynb notebook viewer*
   https://nbviewer.jupyter.org/github/mcsiva/Python/blob/main/Choropleth%20World%20Map%20-%20GDP%20per%20capita-checkpoint.ipynb

2. **Classification HR Analytics - EDA, Model training & prediction**
  *Data Wrangling - imputing NaN values, Feature extraction techniques such as one_hot encoder and Pandas getdummies method used.*
   
   *Decision Tree model* used - *LogRegression (ROC Auc score: 72.2 %), Decision Tree (ROC Auc score: 97.5 %)*
   
   Based on higher roc_auc_score - *Decision tree model* was used to get the *feature importances*:
      City_development Index, Gender, Enrolled University were predicted as important features for Employee Attrition
   Same model is used for *predicting the outcomes* of the test set.
   
   By Applying *Feature selection techniques (ANOVA, Recursive Feat selection)*- we have reduced the number of features from 18 to 10 significant features based on *p-values.*
   *Decision tree classifier* model performance has increase in *accuracy score from (72% to 74.2 %)* and improved *recall score for class 1(attrition) of 44%*
  
   Below is nbviewer link of the *jupyter notebook:*
   https://nbviewer.jupyter.org/github/mcsiva/Python/blob/main/Classification_HR_Analytics.ipynb

3. **Credit Card Fraud Detection**
    *Stratified split of train_test data set* was used as data is highly imbalanced *(<0.1% Class 1 target var value counts)*
   *model training and score* done initially with *Outlier detection model - Isolation Forest*
   Based on scores less than 50% recall rate - Further models are built.
   *XGBoost classifier* - achieved *99.9% accuracy with 76% recall rate* for Class 1 (Fraud transaction class)
   *RandomForest classifier* - achieved *99.9% accuracy with 74% recall rate* for Class 1 (Fraud transaction class)

4. **MNIST image classifier**
   Following Models were trained, tuned and scored for their performance on *MNIST digit image multiclass classification*.
   *RandomForest classifier, SVM Classifier(SVC), Stochastic Grad Classifier,
   TPOT Classifier 2 iterations (Log Reg clssifier and Extra trees classifier), XGBoost classifier with finetuning*
  
   *Model selection for MNIST dataset:*
      a Various models have been considered and performance measure have been done with accuaracy score and mean TPR
      b Based on performance metrics *SVM Classifier and XGB Classifier have acc_score of (97.2%, 97.09%), mean_TPR(97.2%,97.0.9%)* respectively.
      c *Random Forest classifier* - though *accuracy is higher (99.8%), mean TPR is lower (96.1%)*
      d Based on higher performance score for SVM on both accuracy and mean TPR compared to XGB --> *SVM classifier* performs better.
      e *XGB model* based on hyperparameter finetuning achieved *97% accuracy with 97.1% mean TPR*. (marginal improvemnt after finetuning)
      f *SVM classifier* was used to predict the result with final test data.

5. **Product Segmentation** - Model building in *Azure ML Studio* -
   *Twoclass BoostedTree with Onevsall Multiclass classifier* and performance comparison with other models.
   Refer below link to *Microsoft Azure ML model* comparison published in *Azure ML Gallery*
   https://gallery.cortanaintelligence.com/Experiment/MultiClass-label-prediction-for-product-segment-Kaggle-Dataset

6. **Bank Deposit marketing campaign** - Data Analysis and *Model train - ***Performance comparison of class1 prediction - RF and XGB Classifier***
   - *Pandas profiling* , *Pearson coefficient correlation heatmap*, *Label encoding and ordinal (ranked) encoding* applied and feature correlation inference done.
   - The data is related with direct marketing campaigns of a Portuguese banking institution.
   - Features are related to customer credit status, education, job background, previous marketing detail campaigns and current campaign detail with Target outcome for current campaign deposit ('Y' or 'N').
   - *target variable:* Marital status, education, previous campaign ('previous') outcome and A/c Balance are highly correlated with each other (atleast 0.05) -  which can be applied for feature selection to build model for prediction and measure performance.
   - we have calculated confusion matrix of RFClassifier class1 prediction and XGBClassifier class1 prediction
   - The McNemars test results in p-value of 0.45, which is signifacntly higher than considered alpha = 0.05 
   - so we cannot reject the Null Hypothesis of similarity in Model performances
   - Below link for viewing the notebook incase not visible in github:
   - https://nbviewer.jupyter.org/github/mcsiva/Python/blob/main/Bank_Deposit_marketing_campaing_dataAnalysis-1.ipynb
   - https://nbviewer.jupyter.org/github/mcsiva/Python/blob/main/Bank_Deposit_marketing_campaing_Model_train_performance.ipynb
