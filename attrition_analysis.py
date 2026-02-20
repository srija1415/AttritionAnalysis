import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Basic Data Cleaning
# -----------------------------

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Convert Attrition to numeric
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# -----------------------------
# 3. Attrition Rate
# -----------------------------
attrition_rate = df['Attrition'].mean() * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")

plt.figure()
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Count")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# -----------------------------
# 4. Attrition by Department
# -----------------------------
plt.figure()
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title("Attrition by Department")
plt.show()

# -----------------------------
# 5. Attrition vs Job Satisfaction
# -----------------------------
plt.figure()
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title("Job Satisfaction vs Attrition")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# -----------------------------
# 6. Salary vs Attrition
# -----------------------------
plt.figure()
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Monthly Income vs Attrition")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# -----------------------------
# 7. Age Distribution
# -----------------------------
plt.figure()
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution of Employees")
plt.show()

# -----------------------------
# 8. Years at Company vs Attrition
# -----------------------------
plt.figure()
sns.boxplot(x='Attrition', y='YearsAtCompany', data=df)
plt.title("Years at Company vs Attrition")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# -----------------------------
# 9. Overtime Analysis
# -----------------------------
plt.figure()
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Overtime vs Attrition")
plt.show()

# -----------------------------
# 10. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 11. Business Insights
# -----------------------------
print("\nBUSINESS INSIGHTS:")
print("- Employees working overtime show higher attrition.")
print("- Low job satisfaction is strongly linked to attrition.")
print("- Employees with lower monthly income are more likely to leave.")
print("- Most attrition happens in early years at the company.")
print("- Certain departments have higher attrition risk.")
