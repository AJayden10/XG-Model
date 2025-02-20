import seaborn as sns
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn

try:
    df = pd.read_csv("xg_model.csv")
    print(df)

except:
    print(f"Error reading the file")


print(df.head())
df.drop('OwnGoal', axis = 1, inplace = True)
#drops all valaues containing own goal datas(own goals dont count for expected goals model)
df.shape
df.is_goal.value_counts()
#prints a boolean value to display if the shot ended up in a goal or not
plt.figure(figsize=(10,6))
sns.scatterplot(x='x', y= 'y', data = df, hue='is_goal', alpha = 1)
plt.title("ShotMap")
plt.show()
#plots a scatterplot which displays a shot map. Blue values represent a goal and yellow values represent a shot which did not end up in a goal
df.fillna(0, inplace= True)
# replaces all null values with a false boolean

df['shot_distance'] = np.sqrt((df['x'] - 100)**2 + (df['y']-50)**2)
#uses euclidian formula to measure the distance of where the shot was taken from the goal
print(df.shot_distance)
print(df['shot_distance'].hist())
plt.title("Shot Distance Distribution")
plt.show()
#displays a histogram to show how frequently a shot was taken from a particular distance
df.period.value_counts()
df = pd.get_dummies(df, columns = ['period', 'Zone'])
df.columns
df.period_FirstHalf.value_counts()
df.period_SecondHalf.value_counts()

#TRAINING AND TESTING THE MODEL USING LOGISTIC REGRESSION
x = df.drop('is_goal', axis = 1)
y= df['is_goal']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=36)
model = LogisticRegression(max_iter= 2000)
model.fit(X_train, y_train) 
y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred_proba[:10]

final_df = X_test.copy()
final_df['goal_probability'] = y_pred_proba
print(final_df.iloc[380])
print(final_df.head())
