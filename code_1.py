import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.linear_model import ElasticNet,Ridge,Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten,LeakyReLU
from tensorflow.keras.optimizers import Adam 
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVC

df1 = pd.read_csv('focused_synthetic_loan_data.csv')
print(df1.loc[:5,:])
df1["LogLoanAmount"] = np.log1p(df1["LoanAmount"])
df1["LogNetWorth"] = np.log1p(df1["NetWorth"])
df1["IncomeToLoanRatio"] = df1["MonthlyIncome"] / (df1["MonthlyLoanPayment"] + 1)
df1['AssetsToLiabilities'] = df1['TotalAssets'] / (df1['TotalLiabilities'] + 1)


X = df1[['TotalDebtToIncomeRatio', 'CreditScore','LoanDuration', 'InterestRate',"IncomeToLoanRatio" ,"LogLoanAmount","LogNetWorth",'MonthlyLoanPayment',
    'BankruptcyHistory','PaymentHistory',"LengthOfCreditHistory",'AssetsToLiabilities',"CheckingAccountBalance","SavingsAccountBalance","PreviousLoanDefaults","CreditCardUtilizationRate"]]
y = df1["RiskScore"]

#'UtilityBillsPaymentHistory',  
# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Classification
X_C=df1[['MonthlyIncome','TotalDebtToIncomeRatio', 'CreditScore','NetWorth','LoanAmount', 'LoanDuration', 'InterestRate', 'MonthlyLoanPayment',
    'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',"TotalAssets"]]
Y_C=df1["LoanApproved"]
XC_train,XC_test,yC_train,yc_test = train_test_split(X_C,Y_C,test_size=0.3,random_state=42)
scalerC=StandardScaler().fit(XC_train)
XC_train=scalerC.transform(XC_train)
XC_test=scalerC.transform(XC_test)


def meta_mlp():
    model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),LeakyReLU(alpha=0.1),
    Dense(64),LeakyReLU(alpha=0.1),
    Dense(32),LeakyReLU(alpha=0.1),
    Dense(1)
])
    model.compile(optimizer=Adam(0.0001),loss='mse')
    return model


rf = RandomForestRegressor(n_estimators=300,random_state=42)
xgb = XGBRegressor(n_estimators=210, learning_rate=0.05, max_depth= 4, random_state=42)
mlp_model=KerasRegressor(model=meta_mlp,epochs=100, batch_size=8, verbose=0)

#Lets skip to the good part --> Ensemble Learning :)
base_models=[
    ('rf',rf),('xgb',xgb)]

Stack = StackingRegressor(
    estimators=base_models,
    final_estimator = ElasticNetCV(
    l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
    alphas=np.logspace(-3, 1, 20),
    cv=5,
    random_state=42
),
    passthrough=True
)

"""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
  ('poly', PolynomialFeatures(degree=2, include_bias=False)),
  ('scaler', StandardScaler()),
  ('stack', StackingRegressor(
      estimators=[
        ('rf',RandomForestRegressor(n_estimators=300,random_state=42)),
        ('xgb', XGBRegressor(n_estimators=210, learning_rate=0.05, max_depth= 4, random_state=42))
      ],
      final_estimator=ElasticNetCV(
    l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
    alphas=np.logspace(-3, 1, 20),
    cv=5,
    random_state=42
),
      passthrough=True
  ))
])
param_grid = {
  'stack__rf__n_estimators': [100,200],
  'stack__xgb__max_depth': [3,5],
  'stack__mlp__epochs': [50,100]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')"""
Stack.fit(X_train, y_train)

#best_pred = grid.predict(X_test)

mlp_model.fit(X_train, y_train)
pred1 = Stack.predict(X_test)
pred2= mlp_model.predict(X_test).flatten()

# Final Ensemble (Simple Average)
best_pred = (pred1 + pred2) / 2

print(X_test)
print(best_pred)

print("MSE:", mean_squared_error(y_test, best_pred))
print("R² Score:", r2_score(y_test, best_pred))


def plot_graph():
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, best_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Risk Score")
    plt.ylabel("Predicted Risk Score")
    plt.title("Actual vs Predicted Risk Score")
    plt.grid(True)
    plt.show()
plot_graph()

#Classification Begins
model=SVC(kernel='linear',C=0.5,gamma="scale")
model.fit(XC_train, yC_train)

class_preds=model.predict(XC_test)
print(class_preds)
print("Accuracy score: ", accuracy_score(yc_test,class_preds))


from sklearn.metrics import confusion_matrix
import seaborn as sns
def plot_heatmap():
    cm = confusion_matrix(yc_test, class_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

plot_heatmap()

from sklearn.metrics import precision_recall_curve, average_precision_score
y_probs = model.decision_function(XC_test)

def plot_precision():
    precision, recall, _ = precision_recall_curve(yc_test, y_probs)
    avg_precision = average_precision_score(yc_test, y_probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'Avg Precision = {avg_precision:.2f}', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

plot_precision()
import pickle

# Save stacking regressor
with open('stack_model.pkl', 'wb') as f:
    pickle.dump(Stack, f)

# Save KerasRegressor (MLP model)
with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp_model, f)

# Save classifier model (Logistic Regression)
with open('classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the regression scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the classification scaler
with open('scaler_class.pkl', 'wb') as f:
    pickle.dump(scalerC, f)

print("All models and scalers saved successfully.")
