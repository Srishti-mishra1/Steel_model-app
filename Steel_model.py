#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#import shap


# In[3]:


df = pd.read_csv('NIMS_like_fatigue_dataset.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df.describe


# In[10]:


#  Correlation Matrix 
plt.figure(figsize=(18,12))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()


# In[11]:


# Pairplots for Key Features
sns.pairplot(df[['C (%)', 'Mn (%)', 'Cr (%)', 'Ni (%)', 'Ultimate Tensile Strength (MPa)', 'Yield Strength (MPa)']])
plt.suptitle('Pairplot of Key Composition Elements vs Strength', y=1.02)
plt.show()


# In[12]:


# 6. Distribution Plot of Target (Fatigue Strength)
sns.histplot(df['Fatigue Strength (MPa)'], kde=True)
plt.title('Distribution of Fatigue Strength')
plt.show()


# In[13]:


# 7. Boxplot for Carbon Content
sns.boxplot(x=df['C (%)'])
plt.title('Boxplot of Carbon Content')
plt.show()


# In[14]:


# 8. Scatterplot of Carbon vs Fatigue Strength
plt.scatter(df['C (%)'], df['Fatigue Strength (MPa)'])
plt.xlabel('Carbon %')
plt.ylabel('Fatigue Strength (MPa)')
plt.title('Carbon Content vs Fatigue Strength')
plt.show()


# In[15]:


# 9. Feature Engineering - Add Carbon Equivalent (Ceq)
df['Ceq'] = df['C (%)'] + df['Mn (%)']/6 + (df['Cr (%)'] + df['Mo (%)'])/5 + (df['Ni (%)'] + df['Cu (%)'])/15
print(df[['C (%)', 'Mn (%)', 'Cr (%)', 'Mo (%)', 'Ni (%)', 'Cu (%)', 'Ceq']].head())


# In[16]:


# 10. Create Binary Features (High Carbon, High Alloy)
df['High Carbon'] = np.where(df['C (%)'] > 0.6, 1, 0)
df['High Alloy'] = np.where((df['Cr (%)'] + df['Mo (%)']) > 1, 1, 0)


# In[17]:


# 11. Define Features and Target
X = df.drop(['Fatigue Strength (MPa)'], axis=1)
y = df['Fatigue Strength (MPa)']


# In[18]:


# 12. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# 13. Train Multiple Models 📈

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)


# In[20]:


# Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth=10, random_state=42)
dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


# In[21]:


# 14. Evaluation Metrics 📈

def eval_metrics(y_true, y_pred, model_name):
    print(f"---{model_name}---")
    print('R2 Score:', r2_score(y_true, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_true, y_pred)))
    print('MAE:', mean_absolute_error(y_true, y_pred))
    print()

eval_metrics(y_test, y_pred_linreg, 'Linear Regression')
eval_metrics(y_test, y_pred_ridge, 'Ridge Regression')
eval_metrics(y_test, y_pred_dtr, 'Decision Tree Regressor')
eval_metrics(y_test, y_pred_rfr, 'Random Forest Regressor')
eval_metrics(y_test, y_pred_xgb, 'XGBoost Regressor')


# In[22]:


# 15. Compare Predicted vs Actual 
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_rfr, alpha=0.7)
plt.xlabel('Actual Fatigue Strength')
plt.ylabel('Predicted Fatigue Strength')
plt.title('Random Forest: Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()


# In[23]:


# 16. Feature Importance from Random Forest 
importances = rfr.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances - Random Forest")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[24]:


# 17. SHAP Explainability (optional but amazing) 
#explainer = shap.TreeExplainer(rfr)
#shap_values = explainer.shap_values(X_test)

#shap.summary_plot(shap_values, X_test)

# 18. Model Comparison Table 📋
models = ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
r2_scores = [r2_score(y_test, y_pred_linreg), r2_score(y_test, y_pred_ridge),
             r2_score(y_test, y_pred_dtr), r2_score(y_test, y_pred_rfr), r2_score(y_test, y_pred_xgb)]

plt.figure(figsize=(10,5))
sns.barplot(x=models, y=r2_scores)
plt.title('Model R2 Score Comparison')
plt.ylabel('R2 Score')
plt.show()


# In[25]:


from sklearn.model_selection import cross_val_score

# Random Forest 5-Fold Cross Validation for R2 Score
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
cv_r2_scores = cross_val_score(rfr, X, y, cv=5, scoring='r2')  # 5 folds

print("Random Forest 5-Fold CV R2 Scores:", cv_r2_scores)
print("Mean R2 Score:", np.mean(cv_r2_scores))


# In[26]:


from sklearn.preprocessing import PolynomialFeatures

# Create Polynomial Features up to degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("New Feature Shape (after Polynomial Expansion):", X_poly.shape)

# Now you can train models on X_poly instead of X!
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Example: Train Random Forest on Polynomial Features
rfr_poly = RandomForestRegressor(n_estimators=100, random_state=42)
rfr_poly.fit(X_train_poly, y_train_poly)
y_pred_poly = rfr_poly.predict(X_test_poly)

# Evaluate
print("Random Forest (Polynomial Features) R2:", r2_score(y_test_poly, y_pred_poly))


# In[27]:


from sklearn.model_selection import learning_curve

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='r2', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("R2 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Call the function for Random Forest
plot_learning_curve(RandomForestRegressor(n_estimators=100, random_state=42), X, y, title="Random Forest Learning Curve")


# In[28]:


from sklearn.model_selection import validation_curve

param_range = np.arange(1, 21)
train_scores, test_scores = validation_curve(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X, y, param_name="max_depth", param_range=param_range,
    cv=5, scoring="r2", n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(param_range, train_scores_mean, label="Training score")
plt.plot(param_range, test_scores_mean, label="Validation score")
plt.xlabel("Max Depth")
plt.ylabel("R2 Score")
plt.title("Validation Curve for Random Forest (max_depth)")
plt.legend()
plt.grid()
plt.show()


# In[29]:


residuals = y_test - y_pred_rfr

plt.scatter(y_pred_rfr, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.grid()
plt.show()


# In[30]:


plt.figure(figsize=(8,6))
sns.regplot(x=y_test, y=y_pred_rfr, line_kws={"color": "red"})
plt.xlabel("Actual Fatigue Strength")
plt.ylabel("Predicted Fatigue Strength")
plt.title("Prediction Error Plot - Random Forest")
plt.grid()
plt.show()


# In[31]:


# Encode categorical columns if any
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define target and features
target_column = 'Fatigue Strength (MPa)'
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance (XGBoost Model on Fatigue Strength)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# In[37]:


import joblib

# Assuming your trained model objects are named like this:
# If not, replace the variable names accordingly.

# Save the mechanical property prediction model
joblib.dump(model, 'Steel_model.pkl')
print("✅ Steel_model.pkl saved successfully.")


# In[38]:


loaded_mechanical_model = joblib.load('Steel_model.pkl')


# In[ ]:




