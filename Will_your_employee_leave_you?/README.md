# Hackerearth_ML_challenge_will_your_employee_leave_you?
32 rank out of 5164 participent, 
Linear Regression, MySQL Database</br>

## Problems:

### 1. Predict the employee attrition rate in organizations

   - **Problem statement**
   
       Employees are the most important part of an organization. Successful employees meet deadlines, make sales, and build the brand through positive customer interactions.</br>
       
       Employee attrition is a major cost to an organization and predicting such attritions is the most important requirement of the Human Resources department in many organizations. In this problem, your task is to predict the attrition rate of employees of an organization.</br>
       
       [Dataset](https://www.kaggle.com/webman19/hacker-earth-will-your-employees-leave-you)  </br>
       
 ### Solution by using Multiple Linear Regression
 - ANN
 - XGBRegressor
 - statsmodels
 - RandomForestRegressor
 - RidgeCV
 - Lesso
 - BayesianRidge, ARDRegression, LinearRegression
 - SVR


### 2. Employee details

   - **Problem statement**
   
       Find the employee ID and name of all the female employees who work in the Sales department whose:</br>
        - Time since the last promotion has exceeded 1 year
        - Pay scale is above 4.0.</br>
        
       You must order your answer by employee's name.</br>
       
   - **Table Description**</br>
             **Employee**
     | Column  | Type |
     | ------------- | ------------- |
     | Employee_ID  | varcahr(10)  |
     | Age  | int  |
     | Education | int |
     | Relationship_Status | varchar(10) |
     | Hometown | varchar(10) |
     | Name | varchar(30) |
     | Gender | varchar(1) |
     
     **Service**
     | Column  | Type |
     | ------------- | ------------- |
     | Employee_ID  | varcahr(10)  |
     | Unit  | varchar(10)  |
     | Post_Level | int |
     | Time_since_promotion | int |
     | Time_of_service | float |
     | Pay_Scale | float |
     | Collaboration_and_Teamwork	 | int |
     | Compensation_and_Benefits | varchar(10) |
     ### Solution by using MYSQL

#### Accuracy 81.318
