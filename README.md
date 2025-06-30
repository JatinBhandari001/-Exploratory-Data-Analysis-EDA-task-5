# -Exploratory-Data-Analysis-EDA-task-5

## Objective
This project aims to perform **Exploratory Data Analysis (EDA)** on the Titanic dataset using Python. The goal is to extract meaningful insights, identify patterns, understand data distribution, and prepare the dataset for future modeling tasks.

---

## Tools & Technologies Used

- **Python**
- **Pandas** – for data manipulation
- **Seaborn & Matplotlib** – for data visualization
- **Jupyter Notebook** – for analysis and documentation

---

## Dataset

**Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

- 891 records
- Key columns: `Survived`, `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, etc.
- Some missing data in `Age`, `Embarked`, and `Cabin`

---

## Key Analyses Performed

###  Univariate Analysis
- Age distribution using histograms
- Survival counts
- Fare outliers using boxplots

### Bivariate Analysis
- Gender vs Survival
- Passenger class vs Survival
- Scatter plot of Age vs Fare colored by survival

###  Multivariate Analysis
- Pairplot of Age, Fare, Pclass vs Survived
- Correlation heatmap to identify relationships

### Data Cleaning
- Filled missing `Age` and `Embarked` values
- Dropped `Cabin` due to excessive missing data
- Checked and treated skewness in `Fare` using log transformation

---

##  Summary of Insights

- Majority of passengers did not survive (~62%)
- Females had significantly higher survival rates
- First-class passengers were more likely to survive
- Fare had a strong correlation with survival and class
- Data showed right skewness in age and fare
- 
---

##  How to Run

1. Clone this repo
2. Open `Titanic_EDA.ipynb` in Jupyter Notebook or VS Code
3. Make sure the dataset is present in the correct path
4. Run cells sequentially to explore

---

##  Author

**Jatin Bhandari**  
Aspiring Data Analyst | Passionate about insights & visualization  
[LinkedIn] ((https://www.linkedin.com/in/jatin-bhandari-b49173364/))

---
