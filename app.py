
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
# Load the dataset
df = pd.read_csv("Provisional_COVID-19_Deaths_by_Sex_and_Age.csv")

#Basic info

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe(include='all'))


# Check for missing values
print(df.isnull().sum())

#Data cleaning
for col in df.columns:
    if df[col].dtype == 'object':
        # Fill string (object) columns with mode
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    else:
        # Fill numeric columns with integer mean
        mean_val = int(df[col].mean())
        df[col] = df[col].fillna(mean_val)

df = df.drop_duplicates()

# Drop unnecessary columns
df.drop(columns=["Year","Month","Footnote"], inplace=True)
print(df.shape)


# Check unique values in categorical columns
print(df['Sex'].unique())
print(df['Age Group'].unique())

# Final check for missing values
print(df.isnull().sum())



#visualization


pivot_table = df.pivot_table(values='COVID-19 Deaths', index='Age Group', columns='Sex', aggfunc='sum')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlOrBr")
plt.title("Heatmap of COVID-19 Deaths by Age Group and Sex")
plt.ylabel("Age Group")
plt.xlabel("Sex")
plt.tight_layout()
plt.show()


df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['Year'] = df['Start Date'].dt.year
yearly_df = df.groupby('Year')['COVID-19 Deaths'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly_df, x='Year', y='COVID-19 Deaths', marker='o')
plt.title("Yearly COVID-19 Deaths Over Time")
plt.ylabel("Total Deaths")
plt.grid(True)
plt.tight_layout()
plt.show()

stacked_df = df.groupby('Age Group')[['COVID-19 Deaths', 'Pneumonia Deaths', 'Influenza Deaths', 'Pneumonia and COVID-19 Deaths']].sum().reset_index()
stacked_df.set_index('Age Group').plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
plt.title("Death Type Breakdown by Age Group")
plt.ylabel("Number of Deaths")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


death_corr = df[['COVID-19 Deaths', 'Pneumonia Deaths', 'Influenza Deaths', 'Pneumonia and COVID-19 Deaths']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(death_corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Death Types")
plt.tight_layout()
plt.show()


top_states = df.groupby('State')['COVID-19 Deaths'].sum().sort_values(ascending=False).head(10).reset_index()
fig = px.bar(top_states, x='State', y='COVID-19 Deaths', color='COVID-19 Deaths', title="Top 10 States by COVID-19 Deaths")
fig.show()
sex_dist = df.groupby('Sex')['COVID-19 Deaths'].sum().reset_index()

# Plot using matplotlib to create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sex_dist['COVID-19 Deaths'], labels=sex_dist['Sex'], autopct='%1.1f%%', startangle=90)
plt.title('COVID-19 Death Distribution by Sex')
plt.show()



# Encode 'Sex' column (e.g., Male=0, Female=1)
df['Sex_encoded'] = df['Sex'].map({'Male': 0, 'Female': 1})

# Group and aggregate the data
sex_dist = df.groupby('Sex_encoded')['COVID-19 Deaths'].sum().reset_index()

# Prepare input (X) and output (y)
X = sex_dist[['Sex_encoded']]  # independent variable
y = sex_dist['COVID-19 Deaths']  # dependent variable

# Fit Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot the regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sex_encoded', y='COVID-19 Deaths', data=sex_dist)
plt.plot(X, y_pred, color='red', linestyle='--')
plt.title('Linear Regression: COVID-19 Deaths vs. Sex')
plt.xlabel('Sex (Male=0, Female=1)')
plt.ylabel('COVID-19 Deaths')
plt.show()




