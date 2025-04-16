import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.graphics.mosaicplot import mosaic


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

column_names= ['age',
'workclass',
'fnlwgt',
'education',
'education-num',
'marital-status',
'occupation',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'native-country',
'income']

# Load data
data = pd.read_csv(url, names=column_names, skipinitialspace=True)

# clean data
for column in data.columns:
    data[column] = data[column].apply(lambda x: x if x != '?' else None)

df = pd.DataFrame(data)


# User story1 : relationship between education level and income
sns.boxplot(x='income', y='education-num', data=data)
plt.title('Education vs. Income')
plt.xlabel('Income')
plt.ylabel('Education Number')
plt.show()


# User story2 : relationship between sex, marital-status and income
fig, ax = plt.subplots(figsize=(8,10))
mosaic_plot = mosaic(df, ['sex', 'marital-status', 'income'], ax=ax, labelizer=lambda k: "")
ax.set_title('Income Distribution Based on Sex and Marital Status')
ax.set_yticklabels(ax.get_yticklabels(), rotation=10, fontsize=6)
plt.show()


# User story3 : relationship between occupation and income
count_df = pd.crosstab(data['occupation'], data['income'])
ax = count_df.plot(kind='line', marker='o')
plt.title('Line Plot: Occupation vs. Income')
plt.xlabel('Occupation')
plt.ylabel('Population Count')
plt.legend(title='Income Category')
ax.set_xticks(range(len(count_df.index)))
ax.set_xticklabels(count_df.index, rotation=45, ha='right')
plt.show()


# User story4 : relationship between workclass, capital-gain and income
sns.barplot(data=data, x='workclass', y='capital-gain', hue='income', ci=None)
plt.title('Grouped Bar Plot: Workclass vs. Capital-gain by Income')
plt.xlabel('Workclass')
plt.ylabel('Capital-gain')
plt.xticks(rotation=45, ha='right')
plt.show()


# User story5: relationship between age, hours-per-week and income
fig = px.scatter(data, x='age', y='hours-per-week', marginal_x="histogram", marginal_y="histogram", color='income')
fig.update_layout(xaxis_title='Age', yaxis_title='Hours-per-week')
fig.update_layout(title=dict(text="Relationship between Age, Hours-per-week, and Income", x=0.5))
fig.show()