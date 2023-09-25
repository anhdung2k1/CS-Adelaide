import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, normalize, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
#Read and Print diabes.csv out screen
df = pd.read_csv('diabetes.csv')
print(df.head().to_string())
#Drop Columns
x = df.drop(columns=['Outcome'], axis=1)
x_copy=x.copy()
y = df['Outcome']
y_copy=y.copy()
#Print missing data
print("Print the missing value contains ",df.isnull().sum())
feature_mising_value = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x[feature_mising_value] = x[feature_mising_value].replace(0,np.nan)
print("Print null data",x.isnull().sum())
#Imputer
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
x=imputer.fit_transform(x)
x=pd.DataFrame(x,columns=x_copy.columns)
print(x.isnull().sum())
#outliner
for col in x.columns:
    lower_limit=x[col].mean()-3*x[col].std()
    upper_limit=x[col].mean()+3*x[col].std()
    print(f'lower_limit of {col} is: {lower_limit} ')
    print(f'upper_limit of {col} is: {upper_limit} ')
    for index in range(len(x.index) - 1):
        if x.loc[index,col] > upper_limit or x.loc[index,col] < lower_limit:
            x.loc[index, col] = np.nan
    num_outlier = x[col].isnull().sum()
    print(f'num of outlier for {col} is: {num_outlier}')
imputer=SimpleImputer(missing_values=np.nan,strategy='median')
x=imputer.fit_transform(x)
x=pd.DataFrame(x,columns=x_copy.columns)
#Feature Extraction
x['Pregnancies/Age']= x['Pregnancies']/x['Age']

x.loc[(x['BMI']<18.5), 'BMI_Range'] = "underweight"
x.loc[(x["BMI"]>= 18.5) & (x['BMI']<24.9), 'BMI_Range'] = "HealthyWeight"
x.loc[(x["BMI"]>=24.9) & (x['BMI']<29.9), 'BMI_Range']= "overweight"
x.loc[(x["BMI"]>=29.9), 'BMI_Range']= 'obese'

x.loc[(x['Age']<25), 'Age_Range']= "young"
x.loc[(x["Age"]>= 25) & (x['Age']<40), 'Age_Range']= "middle"
x.loc[(x["Age"]>=40), 'Age_Range']= 'old'

x.loc[(x["Glucose"] < 70), 'Glucose_Range'] ="Hipoglisemi"
x.loc[(x["Glucose"] >= 70) & (x['Glucose'] < 100) , 'Glucose_Range'] ="Normal"
x.loc[(x["Glucose"] >= 100) & (x['Glucose'] < 125) , 'Glucose_Range'] ="Imparied_Glucose"
x.loc[(x["Glucose"] >= 125), 'Glucose_Range'] ="Hiperglisemi"

x['BMI/Glucose']= x['BMI']/x['Glucose']
x['Insulin/Glucose']= x['Insulin']/x['Glucose']

x_extraction= x
print("x_extraction\n",x_extraction)
#Encoding
orinal_encoder = OrdinalEncoder()
x[['BMI_Range', 'Age_Range', 'Glucose_Range']]=orinal_encoder.fit_transform(x[['BMI_Range', 'Age_Range', 'Glucose_Range' ]])
print("Orinal Encoder\n",x)
#Normalization
x=normalize(x,norm='l1',axis=0)
x=pd.DataFrame(x,columns=x_extraction.columns)
print("Normalization\n",x)
#Split the test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#Standarization
scaler_ti = StandardScaler()
x_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = scaler_ti.fit_transform(x_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
x_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = scaler_ti.fit_transform(x_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])

ppn_clf = Perceptron(eta0=0.001,max_iter=100,random_state=1)
ppn_clf.fit(x_train,y_train)
cross_val_score(ppn_clf, x_train, y_train, cv=3, scoring='accuracy')
y_pred = ppn_clf.predict(x_test)
y_score_ppn = cross_val_predict(ppn_clf, x_train, y_train, cv=3, method='decision_function')
report = classification_report(y_test,y_pred)
print("Report",report)

print(f'roc_auc_score of perceptron {roc_auc_score(y_train, y_score_ppn)}')
print(f'accuracy score of perceptron {accuracy_score(y_test,y_pred)}')