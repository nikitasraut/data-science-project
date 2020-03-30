
#### importing some modules

import pandas as pd
import numpy as np

train = pd.read_csv(r"D:\R Course\loan prediction dataset\train_ctrUa4K.csv")

data_types = pd.Series(train.dtypes , name = 'type')

only_nums = data_types[data_types!='object']
only_cats = data_types[data_types=='object']

nums_only_df = train.loc[:,list(only_nums.index)]
cats_only_df = train.loc[:,list(only_cats.index)]

## categorical imputation 
from sklearn.impute import SimpleImputer
 
cat_imp = SimpleImputer(strategy = 'most_frequent') 
cat_imputed = cat_imp.fit_transform(cats_only_df)

#### numerical imputation

num_imp = SimpleImputer(strategy = 'median')
num_imputed = num_imp.fit_transform(nums_only_df) 

cat_imp_df = pd.DataFrame(cat_imputed , columns = cats_only_df.columns)
num_imp_df = pd.DataFrame(num_imputed , columns = nums_only_df.columns)

train_df = pd.concat([cat_imp_df ,num_imp_df], axis = 1 )

train_dum_df = pd.get_dummies(train_df.iloc[:,1:] , drop_first = True)


###############################Test data############################

test = pd.read_csv(r"D:\R Course\loan prediction dataset\test_lAUu6dG.csv")

data_types_test = pd.Series(test.dtypes , name = 'type')

only_nums_test = data_types_test[data_types!='object']
only_cats_test = data_types_test[data_types=='object']

nums_only_df_test = test.loc[:,list(only_nums_test.index)]
cats_only_df_test = test.loc[:,list(only_cats_test.index)]

## categorical imputation 
from sklearn.impute import SimpleImputer
 
cat_imp_test = SimpleImputer(strategy = 'most_frequent') 
cat_imputed_test = cat_imp_test.fit_transform(cats_only_df_test)

#### numerical imputation

num_imp_test = SimpleImputer(strategy = 'median')
num_imputed_test = num_imp_test.fit_transform(nums_only_df_test) 

#####

cat_imputed_test_df = pd.DataFrame(cat_imputed_test , columns = cats_only_df_test.columns)
num_imputed_test_df = pd.DataFrame(num_imputed_test , columns = nums_only_df_test.columns)

test_df = pd.concat([cat_imputed_test_df , num_imputed_test_df] , axis = 1)

test_df_dum = pd.get_dummies(test_df.iloc[:,1:] , drop_first = True)


########### ML Technique #################

## 1. SVM(Radial) 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report , roc_auc_score , accuracy_score , confusion_matrix
from sklearn.svm import SVC

X = train_dum_df.iloc[:,0:14]
y = train_dum_df.iloc[:,14]

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.3 ,
                                                       random_state = 2019 , stratify =y)


svc = SVC(probability = True , kernel = 'rbf')
fit_svc = svc.fit(X_train , y_train)
pred_svc = fit_svc.predict(X_test)

print(confusion_matrix(y_test , pred_svc))
print(classification_report(y_test , pred_svc))
print(accuracy_score(y_test , pred_svc))

pred_svc_prob = fit_svc.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test , pred_svc_prob))

y_pred_test = fit_svc.predict(test_df_dum)
y_test_1 = np.where(y_pred_test == 1 ,"Y" , "N")

##########################
TestID=test["Loan_ID"]
Loan_ID=pd.DataFrame(TestID)
Loan_ID=Loan_ID['Loan_ID']
Loan_Status=y_test_1

submit1=pd.DataFrame({'Loan_ID':Loan_ID,'Loan_Status':Loan_Status})

submit1.to_csv(r"C:\Users\user-10\Desktop\loan submision\sub_loan.csv",index=False)

              
########Catboost 
import catboost
from catboost import CatBoostClassifier 

model = CatBoostClassifier(iterations=10)
model_fit=model.fit(X_train,y_train)
preds_class = model_fit.predict(X_test)
print(accuracy_score(y_test,preds_class))

pred_cb_prob = model_fit.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test , pred_cb_prob))

test_pred_cb=model_fit.predict(test_df_dum)        
y_test_2 = np.where(test_pred_cb == 1 ,"Y" , "N")

##########################
TestID=test["Loan_ID"]
Loan_ID=pd.DataFrame(TestID)
Loan_ID=Loan_ID['Loan_ID']
Loan_Status=y_test_2

submit2=pd.DataFrame({'Loan_ID':Loan_ID,'Loan_Status':Loan_Status})

submit2.to_csv(r"C:\Users\user-10\Desktop\loan submision\sub_loan_cb.csv",index=False)

###############################Random Forest 

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=2019,
                                  n_estimators=500,oob_score=True)
mrf=model_rf.fit( X_train , y_train )
y_pred_rf = mrf.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_rf))

pred_rf_prob = mrf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test , pred_rf_prob))

test_pred_rf=mrf.predict(test_df_dum)        
y_test_3 = np.where(test_pred_rf == 1 ,"Y" , "N")

##########################
TestID=test["Loan_ID"]
Loan_ID=pd.DataFrame(TestID)
Loan_ID=Loan_ID['Loan_ID']
Loan_Status=y_test_3

submit3=pd.DataFrame({'Loan_ID':Loan_ID,'Loan_Status':Loan_Status})

submit3.to_csv(r"C:\Users\user-10\Desktop\loan submision\sub_loan_rf.csv",index=False)

#############XgBoost

from xgboost import XGBClassifier
clf = XGBClassifier(random_state=2019)
clf.fit(X_train,y_train)

y_pred_xb = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred_xb))
print(classification_report(y_test, y_pred_xb))
print(accuracy_score(y_test,y_pred_xb))

pred_xb_prob = clf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test , pred_xb_prob))

test_pred_xb=clf.predict(test_df_dum)        
y_test_4 = np.where(test_pred_xb == 1 ,"Y" , "N")

##########################
TestID=test["Loan_ID"]
Loan_ID=pd.DataFrame(TestID)
Loan_ID=Loan_ID['Loan_ID']
Loan_Status=y_test_4

submit4=pd.DataFrame({'Loan_ID':Loan_ID,'Loan_Status':Loan_Status})

submit4.to_csv(r"C:\Users\user-10\Desktop\loan submision\sub_loan_xb.csv",index=False)

##############hist gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, 
                                 max_features=2, max_depth=2, random_state=2019)
gbc_fit=gbc.fit(X_train,y_train)

y_pred_gbc = gbc_fit.predict(X_test)
print(confusion_matrix(y_test, y_pred_gbc))
print(classification_report(y_test, y_pred_gbc))
print(accuracy_score(y_test,y_pred_gbc))

y_pred_gbc = gbc_fit.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_gbc))

test_pred_gbc=gbc_fit.predict(test_df_dum)        
y_test_5 = np.where(test_pred_gbc == 1 ,"Y" , "N")

##########################
TestID=test["Loan_ID"]
Loan_ID=pd.DataFrame(TestID)
Loan_ID=Loan_ID['Loan_ID']
Loan_Status=y_test_5

submit5 = pd.DataFrame({'Loan_ID':Loan_ID,'Loan_Status':Loan_Status})

submit5.to_csv(r"C:\Users\user-10\Desktop\loan submision\sub_loan_gbc.csv",index=False)
