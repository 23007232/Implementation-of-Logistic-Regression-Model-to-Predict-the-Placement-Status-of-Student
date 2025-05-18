# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices. 

## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PAVITHRA P
RegisterNumber: 212223110035
*/
```

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv("/content/Placement_Data (1).csv") 
data.head()
```
![432113561-a346410d-aeec-49ac-aeed-d4bef2d58d1f](https://github.com/user-attachments/assets/f074431b-84e9-4511-9e2d-c3105486e7aa)

```python
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![432113614-7b45405a-b62f-4c9a-abdb-e1bfb91a2f08](https://github.com/user-attachments/assets/46233797-6553-423c-aa39-dbbb6ac443f3)

```python
data1.isnull()
```
![432113955-9b8451a5-d31d-4e56-b62e-46efb65d79ea](https://github.com/user-attachments/assets/25d67ea0-f344-43da-a66a-5372e857d6aa)

```python
data1.duplicated().sum()
```
![432114053-30e31b3d-be9a-4108-a17b-b2f14f879ede](https://github.com/user-attachments/assets/006146b2-f31d-4df9-aba8-baba38bc25fc)
```
le = LabelEncoder()
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in cols:
    data1[col] = le.fit_transform(data1[col])
data1
```
![432114127-f3642867-dad5-4c3a-ac62-095e94562b58](https://github.com/user-attachments/assets/298be6d4-a4bf-42f7-b5c9-2d3600c1e5df)
```python
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
```
![432114274-d687a1f2-6f73-48e5-b06e-d8abe5037ed9](https://github.com/user-attachments/assets/ab7b2fd7-2043-4747-998e-834c17382b9a)
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
```
![432114288-8f99249a-7940-4d0b-a85c-477cf8504063](https://github.com/user-attachments/assets/76f0f9d2-cd51-4331-aa4f-e09868fba60b)

```python
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```
![432114367-45e7d2d5-1f15-47ce-9f86-576dab8adfb3](https://github.com/user-attachments/assets/7943d356-73f5-40a6-b06b-fe3f2d0b50c6)

```python
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```
![432114461-2c2b98fa-bbd3-4439-823a-532a26610e89](https://github.com/user-attachments/assets/c34f5188-5210-4a22-af92-de15e029a0ed)

```python
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
![432114555-fd562591-a3b3-41cc-9bd7-1fc086399fcd](https://github.com/user-attachments/assets/b310d893-bcc0-4918-b292-4fcd8fa617cb)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
