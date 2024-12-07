# %% [markdown]
# # Team 20 
# # Customer Personality Analysis

# %% [markdown]
# ## Introduction
# We are going to use Customer Personality Analysis data set (https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) which contains data of a company’s customers. 
# We will analyze this data set, get insights that help marketting team to conduct better marketing strategies as customizing campaigns for certain customer groups or clusters. 
# 
# ### Question
# How can we identify optimal customer clusters using robust clustering methods, and subsequently tailor marketing strategies for each cluster to drive higher revenue and customer engagement?

# %% [markdown]
# ### Assumptions are:
# 1. **'Graduation'** is the same as **'Bachelor'**
# 2. **'2n Cycle'** is the same as **'Master'**
# 3. **'Income'** more than 125K is an outlier
# 4. **'Age'**: more than 100 years to be ignored

# %%
# Python 3.12.3 was used to compile this file.

# Importing Required Libraries
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import csv


# Define the Location of the dataset CSV file (Comma seperated value)
file_path = r'./data/Project_dataset.csv'    

# Read the dataset 
df = pd.read_csv(file_path)

df.head()

# %%
# Get information on the dataframe and it's columns
df.info()

# %%
# Getting Description on each coloumn and that is:
    # Getting count of values 
    # Getting Mean, standard Deviation, Minimumm, 25% , 50%, 75%, and maximum value
df.describe().T   # T: means transpose 

# %% [markdown]
# **Explanation of variables**:
# 
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# 
# **Products**
# 
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# 
# **Promotion**
# 
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 
# **Place**
# 
# - NumWebPurchases: Number of purchases made through the company’s website
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s website in the last month

# %% [markdown]
# ## Data Exploring, Cleansing & Manipulation

# %% [markdown]
# #### Cleansing **'Income'** values
# Standardize income data by handling missing or erroneous entries to ensure accurate analysis.

# %%
# Cleansing 'Income' values

# find out how many 'Income' values is NA
print('Number of NA values in Income Before cleansing is : ', df['Income'].isna().sum())

# Filter out rows where 'Income' is NA by using .loc and boolean indexing for rows then Dropping those rows from the data frame 'df'. 
df = df.loc[df['Income'].notna()]

print("NA values from 'Income' Removed Completley")

# %%
# Reseting the index to correct index errors after dropping some rows
df.reset_index(drop=True, inplace=True)
df.info()

# %% [markdown]
# #### Checking for **duplicate values** and dropping them if any.
#  Identify and remove duplicate entries to maintain dataset integrity and reliability.

# %%
# Check if there are duplicated values using duplicate() and use sum() function to count how many.
duplicates = df.duplicated().sum()
print('Number of Duplicates in the dataset is: ', duplicates)

# %% [markdown]
# ##### Cleansing and manipulating **Marital_Status** values 
# 1. Dropping 'Absurd' and 'YOLO' values
# 2. Replace 'Alone' by 'Single'
# 3. Replace 'Together' by 'Married'
# 
# Doing this cleansing and manipulating will normalize marital status categories for consistency and meaningful segmentation.

# %%
# Marital_Status Column cleansing:
    # 1. Dropping 'Absurd' and  'YOLO'
    # 2. Replace 'Alone' by 'Single'
    # 3. Replace 'Together' by 'Married' 

# Dropping the rows that has Marital_Status == 'Absurd'
df = df[df['Marital_Status'] != 'Absurd']

# Dropping the rows that has Marital_Status == 'YOLO'
df = df[df['Marital_Status'] != 'YOLO']

# Making Sure All Rows where Marital_Status = 'Absurd'
print('No of Absurd = ',df[df['Marital_Status'] == 'Absurd']['Marital_Status'].count())

# Making Sure All Rows where Marital_Status = 'YOLO' 
print('No of YOLO = ',df[df['Marital_Status'] == 'YOLO']['Marital_Status'].count())

# Revise the Marital_status
df['Marital_Status_revised'] = df['Marital_Status'].replace({
    'Alone': 'Single',    # Replace 'Alone' by 'Single'
    'Together': 'Married' # Replace 'Together' by 'Married'
    })

print('The categories in the Marital Status are: ', df['Marital_Status_revised'].unique())

# Saving the revised marital status in 'Marital_status'
df['Marital_Status']=df['Marital_Status_revised']

# Dropping Marital_Status_revised
df=df.drop(columns=['Marital_Status_revised'])

# Count the number of customers in each Marital Status category
marital_status_counts = df['Marital_Status'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(
    x=marital_status_counts.index, 
    y=marital_status_counts.values, 
    palette='Blues_d'  # Use a corporate-style blue palette
)

# Add labels and title
plt.title('Customer Distribution by Marital Status', fontsize=16, fontweight='bold')
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)

# Add value annotations above bars
for i, value in enumerate(marital_status_counts.values):
    plt.text(
        i, 
        value + 1,  # Position above the bar
        str(value), 
        ha='center', 
        fontsize=10, 
        color='black'
    )

# Customize grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True)  # Remove the left spine for a cleaner look
plt.tight_layout()

# Show the chart
plt.show()

# %%
# Reseting the index to correct index errors after dropping column 'Marital_Status_revised'
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# #### **Education** values Cleansing and Manipulation
# 1. Replace 'Graduation' by 'Bachelor'
# 2. Replace '2n Cycle' by 'Master'
# 
# We will replace the 'Graduation' with 'Bachelor' and combining it with the existing 'Bachelor' class. 
# The '2n Cycle' with 'Master' because the '2n cycle' simply corresponds to graduate level or master's level studies.  We will combine the result with the existing 'Master' class.

# %%
# Education Column cleansing:
# 1. Replace 'Graduation' by 'Bachelor'
# 2. Replace '2n Cycle' by 'Master'

# Replace 'Graduation' by 'Bachelor', and '2n Cycle' by 'Master'
df['Education_revised'] = df['Education'].replace({
    'Graduation': 'Bachelor',
    '2n Cycle': 'Master'
})
print('The categories in the Education are: ', df['Education_revised'].unique())

# Saving the revised Education in 'Education'
df['Education'] = df['Education_revised']

# Dropping the revised Education column
df = df.drop(columns=['Education_revised'])

# Count the number of customers in each Education category
education_counts = df['Education'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(
    x=education_counts.index, 
    y=education_counts.values, 
    palette='Blues_d'  # Use a blue color palette for accessibility
)

# Add labels and title
plt.title('Customer Distribution by Education Level', fontsize=16, fontweight='bold')
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)

# Add value annotations above bars
for i, value in enumerate(education_counts.values):
    plt.text(
        i, 
        value + 1,  # Position above the bar
        str(value), 
        ha='center', 
        fontsize=10, 
        color='black'
    )

# Customize grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True)  # Remove left spine for a cleaner look
plt.tight_layout()

# Show the chart
plt.show()

# %% [markdown]
# #### **Adding** a New Column called **'Age' ( Age of customer)** and **Dropping** any row with customer over 100 years old
# 
# Adding Age of customer will enhance demographic analysis while excluding unrealistic data points.

# %%
# Adding a new column called 'Age'

df['Age'] = 2014 - df['Year_Birth'].astype(int)   # 2014 was the latest year in the dataset

# Finding how many persons with age more than 100 years old
print('Number of persons with Age more than 100 years Before Removing them from our Data is: ', (df[df['Age'] >= 100]['Age'].count()))    # we found 3 persons with Age > 100 years old

# Remove people with age > 100 years
df = df[df['Age'] <= 100]


# Reseting the index to correct index errors after dropping and removing some data
df.reset_index(drop=True, inplace=True)

print('Number of persons with Age more than 100 years After Removing them from our Data is: ', (df[df['Age'] >= 100]['Age'].count()))

# Group ages into 10-year bins and count frequencies
age_bins = pd.cut(df['Age'], bins=range(0, 101, 10), right=False)
age_counts = age_bins.value_counts().sort_index()

# Create a bar chart
plt.figure(figsize=(10, 6))  # Larger figure for clarity
sns.barplot(
    x=age_counts.index.astype(str),  # Convert bin ranges to string for display
    y=age_counts.values,
    palette='Blues_d',  
    edgecolor='black'
)

# Add labels and title
plt.title('Customer Age Distribution (10-Year Ranges)', fontsize=16, fontweight='bold')
plt.xlabel('Age Range (Years)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)

# Add value annotations on each bar
for i, value in enumerate(age_counts.values):
    plt.text(
        i, 
        value + 1,  # Position above the bar
        str(value), 
        ha='center', 
        fontsize=10, 
        color='black'
    )

# Customize the grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True)  # Cleaner look by removing left spine
plt.tight_layout()

# Show the plot
plt.show()

# %% [markdown]
# **Observations**: Most customers are 50 years or younger with majority of the customer base between 40 and 50 years old. In addition, approximately 30% of the customer base being 50 years old or older. Overall, we can see that the customer age is approximately normally distributed, which means it is ultimately helpful in designing broad strategies without excluding significant customer segments.

# %% [markdown]
# #### Visualizing the **'Income'** values.
# 
# Use visual tools to identify trends and irregularities in income distribution.

# %%
# Updated Income Distribution Visualization
plt.figure(figsize=(10, 6))  
# Plot histogram using seaborn
sns.histplot(
    data=df, 
    x='Income', 
    bins=15,  
    kde=True,  
    color='#4E79A7', 
    edgecolor='black', 
    alpha=0.7
)

# Calculate and add mean income line
mean_income = df['Income'].mean()
plt.axvline(mean_income, color='red', linestyle='--', linewidth=1.5, label=f'Mean: ${mean_income:,.2f}')

# Customize the title and labels
plt.title('Distribution of Customer Income', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Income ($)', fontsize=14, labelpad=10)
plt.ylabel('Number of Customers', fontsize=14, labelpad=10)

# Annotate mean income
plt.text(
    mean_income + mean_income * 0.05,  
    plt.gca().get_ylim()[1] * 0.9,   
    f'${mean_income:,.2f}', 
    color='red', 
    fontsize=12, 
    fontweight='bold'
)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Set tick parameters for a polished appearance
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add legend
plt.legend(fontsize=12, loc='upper right', frameon=True, framealpha=0.9)

# Add tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# #### We have a few ***outliers*** in the Income column. 
# Exclude extreme income values to improve dataset reliability and reduce skewness. (assumption **'Income'** more than 125K is an outlier)

# %%
# Example data (replace with your actual DataFrame)
# df = pd.read_csv('your_file.csv')

# Define income range for filtering
income_lower_bound = 10000  # Minimum income threshold
income_upper_bound = 125000  # Maximum income threshold

# Filter the DataFrame based on the defined range
df = df[(df['Income'] >= income_lower_bound) & (df['Income'] <= income_upper_bound)]

# Reset the index after filtering
df.reset_index(drop=True, inplace=True)

# Display the number of rows after filtering
print(f'Number of persons within the range ${income_lower_bound:,}-${income_upper_bound:,}: {len(df)}')

# Define income bins and create labels for the bins
bin_edges = pd.interval_range(start=income_lower_bound, end=income_upper_bound, periods=10)
income_bins = pd.cut(df['Income'], bins=bin_edges, precision=0)
income_counts = income_bins.value_counts().sort_index()

# Format bin labels for clarity
bin_labels = [f"${int(interval.left):,} - ${int(interval.right):,}" for interval in income_counts.index]

# Create a bar chart for income distribution
plt.figure(figsize=(10, 6))
sns.barplot(
    x=bin_labels,  # Use formatted labels for x-axis
    y=income_counts.values,
    palette='Blues_d',  # Professional blue color palette
    edgecolor='black'
)

# Add labels and title
plt.title('Customer Income Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Income Range ($)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)

# Add value annotations on each bar
for i, value in enumerate(income_counts.values):
    plt.text(
        i,
        value + 1,
        str(value),
        ha='center',
        fontsize=10,
        color='black'
    )

# Customize the grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for better readability
sns.despine(left=True)
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# **Observations**: Looking at a glance, there is a wide peak for the income distribution for the customer base in the dataset. We can see that majority of the customer base has a 40k to 70k income, which may suggest a large middle-class customer base, where the customers are relatively equally distributed. This similarity in frequency across this range implies less variability within this income segment, making a more stable, predictable segment of the population. However, we will see when we cluster and segment our data with 

# %% [markdown]
# #### Dropping 2 columns: **'Z_CostContact'** & **'Z_Revenue'**   because they do not add value to the data, 'Z_CostContact'=3 always, 'Z_Revenue'=11 always.
# 
# Remove redundant fields that add no analytical value.

# %%
# Dropping 2 columns: 'Z_CostContact'   &    'Z_Revenue'
    # Reason for Droping 'Z_CostContact' is that column has always a value of 3  
    # Reason for Droping 'Z_Revenue' is that column has always a value of 11  

df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])
print(" Z_CostContact Coloumn Removed")
print(" Z_Revenue Coloumn Removed")
df.head(5)

# %% [markdown]
# #### Changing the format of 'Dt_Customer' to ease calculations
# 
#  Ensure consistent date formatting for temporal analysis.

# %%
# Change the date format
# Convert 'Dt_Customer' to datetime to help in calculations related to date  
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

# %% [markdown]
# #### Add a column called **'Since_Years'** to the data frame  to show how many years has been a customer and Dropping 'Dt_Customer'
# 
# Calculate customer tenure in years for longitudinal insights.

# %%
# Add a colum called 'Since_Years' to the data frame  to show how many years has been a customer
df['Since_Years'] = 2014 - df['Dt_Customer'].dt.year

# Dropping Dt_Customer coloumn since it is not needed after creating Since_Year coloumn
df=df.drop(columns=['Dt_Customer'])
df.head()

# %% [markdown]
# #### Add a column called **'Total Number Purchases'** to the data frame  to show how many number of purchases customer done
# 
# Summarize all purchase records per customer for behavioral analysis.

# %%
# Creating a new column in the df called 'Total Number Purchases' = 'NumWebPurchases' + 'NumStorePurchases' + 'NumStorePurchases'
df['Total Number Purchases'] = df['NumWebPurchases'].astype(int) + df['NumStorePurchases'].astype(int) + df['NumCatalogPurchases'].astype(int)
df.head(5)

# %%
# Scatter plot for Total Number of Purchases by Age
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=df['Age'], 
    y=df['Total Number Purchases'], 
    hue=df['Total Number Purchases'],  
    size=df['Total Number Purchases'],  
    sizes=(20, 200), 
    palette='coolwarm', 
    alpha=0.8,
    edgecolor='black'
)

# Add titles and labels
plt.title('Total Number of Purchases by Age', fontsize=16, fontweight='bold')
plt.xlabel('Age (Years)', fontsize=14)
plt.ylabel('Total Number of Purchases', fontsize=14)

# Add a grid for readability
plt.grid(axis='both', linestyle='--', alpha=0.7)

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Aggregate data by age groups for better visualization
df['Age Group'] = pd.cut(df['Age'], bins=range(20, 81, 10), right=False)
age_group_purchases = df.groupby('Age Group')['Total Number Purchases'].sum()

# Create a bar chart
plt.figure(figsize=(10, 6))

sns.barplot(
    x=age_group_purchases.index.astype(str), 
    y=age_group_purchases.values,
    palette='Blues_d',  
    edgecolor='black'
)

# Add titles and labels
plt.title('Total Purchases by Age Group', fontsize=16, fontweight='bold')
plt.xlabel('Age Group (Years)', fontsize=14)
plt.ylabel('Total Purchases', fontsize=14)

# Annotate values on each bar
for i, value in enumerate(age_group_purchases.values):
    plt.text(
        i, 
        value + 5,  
        f'{value}', 
        ha='center', 
        fontsize=10, 
        color='black'
    )

# Customize grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True)  
plt.tight_layout()

# Show the plot
plt.show()

# %% [markdown]
# Examine purchase trends across different age brackets to uncover patterns.

# %% [markdown]
# #### Add a column called **'Total Amount Purchases'** to the data frame  to show how much in Dollars value the customer done purchases.
# 
# Provide a cumulative view of spending behavior per customer.

# %%
# Creating a new column in the df called 'Total Amount Purchases' = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts']+ df['MntFishProducts']+ df['MntSweetProducts']+ df['MntGoldProds']
df['Total Amount Purchases'] = df['MntWines'].astype(int) + df['MntFruits'].astype(int) + df['MntMeatProducts'].astype(int)+ df['MntFishProducts'].astype(int)+ df['MntSweetProducts'].astype(int)+ df['MntGoldProds'].astype(int)
df.head(5)

# %% [markdown]
# #### Add a column called **'No of Accepted Campaigns'** to the data frame to show how many campains or promotions the customer accepted.
# 
#  Quantify customer engagement in marketing campaigns.

# %%
#Creating a feature to get a sum of accepted promotions 
df["No of Accepted Campains"] = df["AcceptedCmp1"].astype(int)+ df["AcceptedCmp2"].astype(int)+ df["AcceptedCmp3"].astype(int)+ df["AcceptedCmp4"].astype(int)+ df["AcceptedCmp5"].astype(int)
df.tail(5)

# %% [markdown]
# #### Adding a column called **'Family_Size'** to the data frame to show how many members in the family
# 
# Define family composition as a factor for consumption patterns.

# %%
# Adding a column called 'Family_Size' to the data frame to show how many members in the family
df['Family_Size'] = 1 + df['Kidhome'].astype(int) + df['Teenhome'].astype(int) + df['Marital_Status'].apply(lambda x: 1 if x == 'Married' else 0)
df.head()

# %% [markdown]
# #### Making sure that the type of **'Recency'** column is a float type to avoid problems when doing dimensional reduction (PCA)
# 
# Standardize data type for precise calculations.

# %%
# Making sure that the type of recency coloumn is float
df['Recency'] = df['Recency'].astype(float)

# %% [markdown]
# #### Apply one-hot encoding to the **Education** and **Marital_Status** columns to convert categorical data ('Education', 'Marital_Status') into numerical format.
# 
# Convert categorical variables into binary indicators for model compatibility.

# %%
# Applying One-Hot Encoding on the categorical variables ('Education', 'Marital_Status') To Convert them into Numerical values
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=False)
df.head()

# %% [markdown]
# #### Keep a copy of the modified data set before scaling (df_org)
# 
# Save the pre-processed dataset for comparison and validation (as df_org).

# %%
df_org =df.copy()  # Keep a copy of the modified data set before scaling
df_org.head(10)

# %% [markdown]
# #### **PCA** will be applied to the **numerical columns** 
# PCA will be applied to the numerical columns while leaving out the binary categorical variables because doing so would distort the components, reducing the interpretability and effectiveness of PCA. The binary data can then be handled separately through clustering or as features for later analysis.

# %%
# Dividing the columns into two types: 
# 1. Numerical values (in those we are going to standarize the values )
# 2. Categorical or Boolean

# ID of customers Excluded from scaling
# Exluding Totals from PCA 
# Excluding: 'Age',
# Excluding: 'Since_Years','Total Number Purchases', 'Total Amount Purchases',
# Excluding: 'No of Accepted Campains', 'Family_Size'

numerical_all = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
       'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
print('Numerical Coloumns identified')

non_numerical_all = ['Education_Bachelor',
       'Education_Basic', 'Education_Master', 'Education_PhD',
       'Marital_Status_Divorced', 'Marital_Status_Married',
       'Marital_Status_Single', 'Marital_Status_Widow']
print('Non-Numerical Coloumns identified')

# %% [markdown]
# #### We would Standarizing the numerical values (numerical_all coloumns) only using **standardscaler()**.

# %%
# Standarizing the numerical values (numerical_all coloumns)
scaler = StandardScaler()
df[numerical_all] = scaler.fit_transform(df[numerical_all])
df_scaled = df[numerical_all]
print('All Numerical Values has been standarized')
df_scaled.head()

# %% [markdown]
# #### Conduct Dimension reduction (PCA with n_components = 3) get a new dataset where each data point is represented by 3 principal components instead of the original 23 features.
# 
# We use Principal Component Analysis (PCA) to simplify the dataset by combining 23 original features into three key components. This step reduces complexity while retaining most of the data's meaningful patterns, making it easier to analyze and visualize.

# %%
# We need to conduct Dimention reduction (PCA)
# Import the PCA class
from sklearn.decomposition import PCA

scaled_df = df_scaled.copy()
# n_components = 3 means get a new dataset where each data point is represented by 3 principal components (instead of the original 25 features).
pca = PCA(n_components = 3,random_state = 42) 
scaled_df = pca.fit_transform(scaled_df)
pca_data = pd.DataFrame(scaled_df, columns=["Feature1","Feature2", "Feature3"])

print("The PCA transformed dataset is:")
pca_data.head()

# %% [markdown]
# #### Visualizing the new feature set

# %%
# Graph of PCA
from mpl_toolkits.mplot3d import Axes3D

x = pca_data["Feature1"]
y = pca_data["Feature2"]
z = pca_data["Feature3"]

print(f"The dataset after dimensionality reduction :")

# Add a color scale for clarity (optional: you can categorize points if needed)
colors = sns.color_palette("viridis", n_colors=len(pca_data))

# Start the plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with enhancements
scatter = ax.scatter(
    x, y, z,
    c=range(len(pca_data)),  # Add a color scale
    cmap='viridis',  
    s=50,  
    edgecolor='k',  
    alpha=0.8  
)

# Add a color bar to indicate point order
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Data Point Index', fontsize=12)

# Set axis labels
ax.set_xlabel('Principal Component 1 (Feature1)', fontsize=12, labelpad=10)
ax.set_ylabel('Principal Component 2 (Feature2)', fontsize=12, labelpad=10)
ax.set_zlabel('Principal Component 3 (Feature3)', fontsize=12, labelpad=10)

# Add a title
ax.set_title('PCA Dimensionality Reduction: 3D Visualization', fontsize=14, fontweight='bold', pad=20)

# Adjust layout for better readability
plt.tight_layout()

# Show the plot
plt.show()

# %% [markdown]
# # Segmentation
# Determining the optimum number of clusters using Elbow method. Clustering involves grouping similar customers based on their behavior and demographics. The Elbow Method helps determine the best number of groups by analyzing how the clustering quality improves as we increase the number of clusters.

# %%
# Segmentation
# importing KElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer

print("Determining the optimam number of clusters using Elbow method :")
_, axes = plt.subplots(figsize=(10,6))

elbow = KElbowVisualizer(KMeans(), k=8, timings=False, locate_elbow=True, size=(1260,450))
elbow.fit(pca_data)

axes.set_title("\nDistortion Score Elbow For KMeans Clustering\n",fontsize=25)
axes.set_xlabel("\nK",fontsize=20)
axes.set_ylabel("\nDistortion Score",fontsize=20)

sns.despine(left=True, bottom=True) 
plt.show()

# %% [markdown]
# #### From the Distortion Score, and the Elbow method from the above graph **K=4 is the optimal number of clusters**, therefore Kmeans clustering will be conducted using K=4

# %%
# Perform K-means clustering Using K = 4
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(scaled_df)

# Create a new DataFrame
clustered_df = pd.DataFrame(scaled_df, columns=["Feature1", "Feature2", "Feature3"])  
clustered_df['Cluster'] = clusters.labels_

# Display the clustered data
clustered_df.head(10)

# %% [markdown]
# #### Adding the 'Cluster' column to the orignal dataframe 'df_org'.
# 
#  Once clusters are defined, each customer is assigned to a group. This allows us to segment the customer base, making it easier to analyze and tailor strategies for each group.

# %%
#Adding the Clusters Coloumn to the orignal dataframe 'df_org'.
df_org["Cluster"]= clusters.labels_
df_org.head(5)

# %% [markdown]
# #### Visualizing the clusters in 3D
# 
# Visualizing the clusters in three dimensions helps us understand how customers are grouped based on the key components identified by PCA. It provides a clear picture of the relationships between different customer segments.

# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

# Extract X, Y, Z, and Cluster labels
x = clustered_df["Feature1"]
y = clustered_df["Feature2"]
z = clustered_df["Feature3"]
clusters = clustered_df["Cluster"]

# Define a red-green colorblind-friendly palette
palette = ['#1f77b4', '#ff7f0e', '#9467bd', '#7f7f7f']  # Blue, Orange, Purple, Gray

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Loop through each cluster and plot points with the defined color palette
for i in range(clusters.nunique()):  # Dynamically adjust for the number of clusters
    ax.scatter(
        x[clusters == i], 
        y[clusters == i], 
        z[clusters == i], 
        color=palette[i], 
        label=f'Cluster {i}', 
        s=60,  
        edgecolor='k',  
        alpha=0.85  
    )

# Set axis labels with business-like design
ax.set_xlabel('Principal Component 1 (Feature1)', fontsize=10, labelpad=10)
ax.set_ylabel('Principal Component 2 (Feature2)', fontsize=10, labelpad=10)
ax.set_zlabel('Principal Component 3 (Feature3)', fontsize=10, labelpad=10)

# Add a descriptive title
ax.set_title('3D Cluster Visualization of PCA Components', fontsize=16, fontweight='bold', pad=20)

# Add a legend to identify clusters
ax.legend(title='Clusters', fontsize=10, title_fontsize=12, loc='upper left', frameon=True, framealpha=0.9)

# Add gridlines for perspective
ax.grid(True, linestyle='--', alpha=0.6)

# Adjust layout for a clean look
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# #### Identifying the *DOMINANT VARIABLE* in each Feature
# 
# By examining the key characteristics that define each cluster, we can better understand what makes customers in each group unique. This insight enables us to develop targeted strategies, such as personalized marketing campaigns or tailored product offerings.

# %%
pca_components = pd.DataFrame(pca.components_, columns=df_scaled.columns, index=["Feature1", "Feature2", "Feature3"])
print("Loadings of the original variables in the principal components:")
print(pca_components)

# %%
# Identify dominant variables by component
dominant_features = pca_components.apply(lambda x: x.abs().nlargest(3).index, axis=1)
print("Most influential variables by component:")
print(dominant_features)

# %% [markdown]
# **Observation for PCA, clustering/segmentation**: 
# 
# After applying PCA, the data can be visualized in a 3D space, but the visualization remains unclear. Without clustering, the data points appear scattered and disorganized, with colors randomly distributed, making it difficult to discern meaningful patterns. However, after applying clustering with Kmeans clustering method, it is determined that 4 is the ideal number of clusters (k).
# 
# With the clusters identified, the visualization in 3D becomes much clearer, as the data points are now grouped into distinct, well-separated clusters with consistent colors representing each group. This segmentation provides a more structured view of the data, enabling easier interpretation of group patterns.
# 
# Further analysis is conducted by examining the dominant features within each cluster. By analyzing the feature contributions to each cluster, we gain insights into the key characteristics that define each group. 
# 1. Feature 1's top 3 influential variables are 'Income', 'NumCatalogPurchases', and 'MntMeatProducts'
# 2. Feature 2's top 3 influential variables are 'Teenhome', 'NumDealsPurchases', and 'NumWebPurchases'
# 3. Feature 3's top 3 influential variables are 'AcceptedCmp4', 'Response', 'AcceptedCmp2'
# 
#  This approach allows for targeted strategies to be developed, such as tailoring marketing campaigns, optimizing resource allocation, or improving product offerings, based on the distinct attributes of each cluster.

# %%
# Generate the heatmap for PCA component loadings
plt.figure(figsize=(10, 6))  

# Customize the heatmap
sns.heatmap(
    pca_components,
    annot=True,  
    fmt=".2f",  
    cmap="coolwarm",  
    cbar=True,  
    linewidths=0.5,  
    linecolor='black',  
    annot_kws={"fontsize": 10},  
)

# Add a descriptive title
plt.title("Loadings of Original Variables in Principal Components", fontsize=16, fontweight='bold', pad=15)

# Add axis labels
plt.xlabel("Original Variables", fontsize=14, labelpad=10)
plt.ylabel("Principal Components", fontsize=14, labelpad=10)

# Improve layout for a clean presentation
plt.tight_layout()

# Display the heatmap
plt.show()


# %% [markdown]
# The above heatmap shows a more detailed, easy-to-read output of how all the original features contribute to the patterns defined by PCA. Each principal component represents a new, simplified dimension of the data with the heatmap helping the reader to see which original features are most important for each component. Warm colors (red) mean the feature has a strong positive influence, whereas the cool colors (blue) mean it has a strong negative influence. 
# 
# 1. **Feature 1**: has mostly red or strong positive influence from 'Income', all the "Mnt*" products, 'NumWebPurchases', 'NumStorePurchases', and 'NumStorePurchases'. These variables lie approximately at 0.3 importance. On the other hand, two variables have a negative influence on feature 1, namely 'Kidhome' and 'numWebVisitsMonth", which both lie at approximately -0.25 influence. 
# 
# 2. **Feature 2**: has significant positive influence from the 'Teenhome' variable at 0.54 influence and 'NumDealsPurchases' at 0.46 influence. In addition, 'NumWebPurchases' also has some significant positive influence at 0.35. The 'Year_Birth' variable has the most negative influence at -0.35. In addition, there is a noticeable neutral to slightly negative influence for promotion variables 0.1 to -0.18, which suggests that this Feature 2 is classified by not very receptive towards promotions.
# 
# 3. **Feature 3**: has significant positive influence from the Promotion related variables. The 'AcceptedCmp1' and the related variables all have positive influence ranging from 0.2 to 0.42. In addition, there is slight positive influence from 'NumWebVisitsMonth' and 'MntWines' at 0.2. Finally, the other products variables have slight negative influence ranging from -0.06 to -0.2.

# %% [markdown]
# ### **Feature 1 (Principal Component 1): General Spending and Income-Driven Behavior**
# 
# **Analysis**  
# - **Strong Positive Contributions**:  
#   - **Income** and spending on various product categories (`Mnt*` variables).  
#   - Purchases across multiple channels (`NumWebPurchases`, `NumStorePurchases`).  
#   - Likely captures general spending capacity and behavior.  
# - **Higher Values**:  
#   - Indicate customers with higher disposable income and more diverse purchasing habits.  
# - **Negative Contributions**:  
#   - `Kidhome` and `NumWebVisitsMonth` suggest households with more children and frequent website visits are associated with lower general spending.
# 
# **Conclusion**  
# Feature 1 may represent a **"High-Income and High-Spending Segment"**, contrasting with households that are cost-sensitive or prioritize web-based browsing over purchases.
# 
# ---
# 
# ### **Feature 2 (Principal Component 2): Household-Oriented Behavior and Deals Sensitivity**
# 
# **Analysis**  
# - **Dominated by Positive Influences**:  
#   - `Teenhome` (number of teenagers).  
#   - `NumDealsPurchases` (sensitivity to discounts).  
#   - `NumWebPurchases` (online purchase behavior).  
# - **Dominated by Negative Influences**:  
#   - `Year_Birth` (likely related to older customers).  
#   - Slight resistance to promotions (`AcceptedCmp*`).  
# 
# This suggests a focus on **family-oriented customers** who are:  
# - More likely to buy discounted products, driven by practical and cost-conscious choices.  
# - Possibly younger or middle-aged households, as older customers (negative `Year_Birth`) are less aligned with this feature.
# 
# **Conclusion**  
# Feature 2 may represent a **"Family-Oriented, Discount-Sensitive Segment"**, prioritizing deals and online purchases with less engagement in marketing campaigns.
# 
# ---
# 
# ### **Feature 3 (Principal Component 3): Promotion Responsiveness**
# 
# **Analysis**  
# - **Strong Positive Contributions**:  
#   - Promotion-related variables (`AcceptedCmp1`, ..., `AcceptedCmp5`, and `Response`).  
#   - Highlights customers actively responding to marketing campaigns.  
# - **Slight Positive Contributions**:  
#   - `MntWines` and `NumWebVisitsMonth` suggest minor alignment with premium/niche products and browsing behavior.  
# - **Negative Contributions**:  
#   - Other product categories (`Mnt*`) indicate less general spending outside promotional offers.
# 
# **Conclusion**  
# Feature 3 likely identifies a **"Promotion-Engaged Segment"**, comprising customers highly receptive to campaigns and prioritizing discounts/offers over organic spending.
# 
# ---
# 
# ### **General Observations and Recommendations**
# 
# #### **Segment-Specific Strategies**
# 1. **Feature 1 (High-Income, High-Spending Segment)**  
#    - Target with **premium products** and **personalized offers**.  
#    - Emphasize convenience and exclusivity (e.g., personalized store experiences or loyalty rewards).  
# 
# 2. **Feature 2 (Family-Oriented, Discount-Sensitive Segment)**  
#    - Engage with **family-value bundles**, **discounts**, and targeted online promotions.  
#    - Highlight practical, budget-friendly options.  
# 
# 3. **Feature 3 (Promotion-Engaged Segment)**  
#    - Focus marketing campaigns and seasonal promotions on this segment.  
#    - Use responsiveness to test new product launches or experimental campaigns.  
# 
# #### **Cross-Segment Insights**
# - **Feature 2 (Resistant to Promotions)**: Showcase long-term value rather than immediate discounts.  
# - **Feature 3 (Promotion-Engaged Customers)**: Develop retention strategies to drive spending beyond discounts.  
# 
# #### **Business Implications**
# - **Tailored Marketing**: Allocate resources effectively based on segment insights.  
#   - Invest in online shopping infrastructure for Feature 2.  
#   - Develop frequent campaigns to retain Feature 3 customers.  
#   - Enhance in-store experiences for Feature 1 customers.

# %% [markdown]
# ### Clusters Analysis

# %%
# Display the number of data points in each cluster
cluster_counts = clustered_df['Cluster'].value_counts()
print("Number of data points in each cluster:")
print(cluster_counts)
print("\n")

# Calculate mean for each cluster
print("Mean for each feature for each cluster:")
cluster_summary = clustered_df.groupby('Cluster').mean()  
print(cluster_summary)
print("\n")

# Calculate standard deviation for each feature in each cluster
print("Standard deviation of features for each cluster:")
cluster_std = clustered_df.groupby('Cluster').std()
print(cluster_std)
print("\n")

# %%
# Display the number of customers in each cluster
cluster_counts = clustered_df["Cluster"].value_counts().sort_index()
cluster_labels = cluster_counts.index  
cluster_values = cluster_counts.values  

# Define a red-green colorblind-friendly palette
colors = ['#4E79A7', '#F28E2B', '#9467BD', '#7F7F7F']  #

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(cluster_labels, cluster_values, color=colors, edgecolor='black', alpha=0.85)

# Add chart title and axis labels
plt.title("Number of Customers in Each Cluster", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Clusters", fontsize=14, labelpad=10)
plt.ylabel("Count of Customers", fontsize=14, labelpad=10)

# Annotate the bars with data values
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, height + 20,  
        f'{int(height)}',                               
        ha='center', va='bottom', fontsize=12, color='black', fontweight='bold'
    )

# Customize gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(cluster_labels, fontsize=12)  
plt.yticks(fontsize=12)                  

# Remove unnecessary borders for a cleaner look
sns.despine(left=True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the chart
plt.show()


# %% [markdown]
# ## Cluster Analysis Conclusions
# 
# ### **1. Cluster Sizes**
# - **Observation:**
#   The number of data points in each cluster varies significantly:
#   - Cluster 3 has the largest size with **964 data points**.
#   - Cluster 2 is the smallest with **145 data points**.
#   - Clusters 1 and 0 have **476** and **587** data points, respectively.
# 
# - **Conclusion:**
#   - **Cluster 3:** Likely represents a broad group of customers with common or average characteristics.
#   - **Cluster 2:** Represents a niche segment, possibly an outlier group or a specialized demographic.
#   - Cluster sizes suggest different levels of customer importance or distinctiveness.
# 
# ---
# 
# ### **2. Mean Feature Values**
# 
# #### **Cluster 0** (balanced spenders, high family-oriented)  
# - Moderate means for `Feature1` (0.25), and high mean for `Feature2` (1.74).  
# - Positive mean for `Feature3` (0.03).  
# - **Interpretation:**  
#   Represents a balanced group, possibly **family-oriented customers** who also respond to some promotional efforts.
# 
# #### **Cluster 1** (moderate spenders, low promotion sensitive, low family-oriented)
# - Moderate positive mean for `Feature1` (3.07).  
# - Slightly negative means for `Feature2` (-0.55) and `Feature3` (-1.17).  
# - **Interpretation:**  
#   Represents customers with moderate spending habits but low promotional responsiveness and family-oriented behavior.
# 
# #### **Cluster 2** (high spenders, promotion sensitive)     
# - High means across all features, especially `Feature1` (4.61) and `Feature3` (3.15).  
# - **Interpretation:**  
#   Represents high-spending, promotion-sensitive customers with distinct purchasing behavior. These customers may be the **most valuable** but also the smallest segment.
# 
# #### **Cluster 3** (average features)
# - Low mean for `Feature1` (-2.37) and `Feature2` (-0.61).  
# - Slightly positive mean for `Feature3` (0.08).  
# - **Interpretation:**  
#   Likely represents customers with low spending/income levels (`Feature1`), moderate sensitivity to household behaviors (`Feature2`), and minimal responsiveness to promotions (`Feature3`).
# ---
# 
# #### **Cluster 0** (Bit more diverse in spending patterns)
# - Similar variability across features, with slightly higher variation in `Feature1` (1.14).  
# - **Interpretation:**  
#   Indicates a relatively diverse group, particularly in spending habits.
# 
# #### **Cluster 1** (Consistent spending patterns, but slightly less variability in promotion engagemnet)
# - Moderate variability, especially in `Feature1` (1.09).  
# - **Interpretation:**  
#   Reflects consistent spending behavior but moderate differences in family-related and promotion engagement.
# 
# #### **Cluster 2** (High varaiability in promotion response = potential for marketing)
# - Highest variability across features, particularly `Feature3` (2.03).  
# - **Interpretation:**  
#   This group is heterogeneous, with diverse spending and promotion response patterns. This could reflect a mix of high-value customers with varying preferences.
# 
# ### **3. Standard Deviation (Variability)**  
# #### **Cluster 3** (Most consistent spending patterns)
# - Lowest variability across features, especially `Feature1` and `Feature3`(0.64).  
# - **Interpretation:**  
#   Indicates a homogeneous group with consistent behavior patterns.
# ---
# 
# ### **Key Business Implications**   
# 1. **Cluster 0:**  
#    Family-oriented customers with balanced traits. Leverage **family-value bundles or targeted online campaigns**.
#    
# 2. **Cluster 1:**  
#    Moderate spending group with low promotional responsiveness. Highlight **long-term value and practical deals** to engage them effectively.
#    
# 3. **Cluster 2:**  
#    Small, high-value, but diverse customers. Consider **personalized, premium offers and exclusive promotions** to maximize returns.
#    
# 4. **Cluster 3:**  
#    Represents a large, consistent group. Focus marketing efforts on broad, cost-effective strategies like **basic loyalty programs or maintaining engagement**.

# %% [markdown]
# #### Clusters analysis based on Education & Marital Status
# 
# By examining how clusters vary across education levels and marital statuses, we can identify patterns and preferences unique to different segments. This analysis helps uncover actionable insights, such as which educational groups are most likely to engage with certain products or services.

# %%
# for Under Standing the Clusters and who is in each cluster

# Define the categorical columns explicitly
categorical_columns = [
       'Education_Bachelor',
       'Education_Basic', 'Education_Master', 'Education_PhD',
       'Marital_Status_Divorced', 'Marital_Status_Married',
       'Marital_Status_Single', 'Marital_Status_Widow'
]

# Remove duplicates from df_org columns (if any)
df_org = df_org.loc[:, ~df_org.columns.duplicated()]

# Ensure the specified categorical columns exist in the DataFrame
valid_categorical_columns = [col for col in categorical_columns if col in df_org.columns]

# Group data by clusters
cluster_groups = df_org.groupby('Cluster')

# Profile categorical data for each cluster
categorical_profiles = {}
for cluster, group in cluster_groups:
    cluster_profile = {}
    for col in valid_categorical_columns:
        # Ensure the column is in boolean format (True/False)
        if group[col].dtype == 'bool':
            # Calculate the percentage distribution for True and False values in the cluster
            value_counts = group[col].value_counts(normalize=True) * 100  
            cluster_profile[col] = value_counts
    categorical_profiles[cluster] = cluster_profile

# Convert profiles to DataFrames for readability
cluster_summaries = {}
for cluster, profile in categorical_profiles.items():
    summary = pd.DataFrame(profile).fillna(0)  # Replace NaNs with 0 for missing categories
    cluster_summaries[cluster] = summary

# Print profiles for each cluster
for cluster, summary in cluster_summaries.items():
    print(f"Cluster {cluster} Profile:")
    print(summary.T)  
    print("-" * 50)

# %% [markdown]
# ###  Marketing Strategies based on Education & Marital Status
# 
# Using the insights gained, we can craft targeted marketing strategies tailored to specific educational or marital demographics. For instance, a campaign could focus on high-income individuals with advanced degrees or families with young children, ensuring relevance and improving campaign success rates.
# 
# 
#  **Cluster 0**:
# - **Profile**: Mostly married individuals, with a significant proportion having Basic Education. Bachelor's degree holders are also nearly evenly distributed.
# - **Strategy**: Target family-oriented campaigns and emphasize value-driven offers.
# 
#  **Cluster 1**:
# - **Profile**: Mostly married with a noticeable singles, with significant proportion having bachelor's degree, with a moderate number of master's and PhD holders.
# - **Strategy**: Promote professional development tools, memberships, or luxury products, and flexible offerings.
# 
#  **Cluster 2**:
# - **Profile**: Highe percentage of married individuals with a notable proportion of singles, with a bachelor and higher education levels (master's and PhD). .
# - **Strategy**: Focus on premium offers and memberships, interested in family-oriented products and career-focused services.
# 
#  **Cluster 3**:
# - **Profile**: Mostly married individuals, with a mix of bachelor's and master's degree holders. Noticeably higher representation of individuals with Basic Education compared to other clusters.
# - **Strategy**: Highlight family-focused solutions and loyalty-based programs, while also addressing the needs of moderately educated individuals.
# 

# %% [markdown]
# #### Visualizing Clusters Vs Education & Marital Status
# 
# Graphical representations make it easier to interpret the relationship between education, marital status, and customer clusters. These visualizations highlight trends and overlaps, helping stakeholders understand key drivers and areas for strategic focus.

# %%
# Visualizing Clusters Vs Education & Marital Status

import pandas as pd
import matplotlib.pyplot as plt

# Group data by clusters
cluster_groups = df_org.groupby('Cluster')

# Create an empty list to collect the data for plotting
plot_data = []

# Profile categorical data for each cluster and collect the True percentage for each category
for cluster, group in cluster_groups:
    for col in valid_categorical_columns:
        if group[col].dtype == 'bool':  # Ensure the column is boolean
            # Calculate the percentage of True values in each cluster
            true_percentage = group[col].mean() * 100  # Mean of booleans gives the percentage of True values
            plot_data.append({'Cluster': cluster, 'Category': col, 'True Percentage': true_percentage})

# Convert the list to a DataFrame
plot_df = pd.DataFrame(plot_data)

# Pivot the data for stacked bar chart
plot_pivot = plot_df.pivot(index='Category', columns='Cluster', values='True Percentage')

# Define a business-style color palette
business_palette = ['#4E79A7', '#76B7B2', '#A0A0A0', '#545454']  

# Plot the stacked bar chart
plot_pivot.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 8),
    color=business_palette,
    edgecolor='black',
    alpha=0.85
)

# Customize the chart
plt.title('Percentage Distribution by Category and Cluster', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=14, labelpad=10)
plt.ylabel('True Percentage (%)', fontsize=14, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Add gridlines
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate bars with percentages
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, color='white')

# Add legend
plt.legend(title='Cluster', fontsize=10, title_fontsize=12, loc='upper right', frameon=True, framealpha=0.9)

# Optimize layout
plt.tight_layout()

# Show the plot
plt.show()



# %% [markdown]
# #### Showing Distribution of Feature1 & Feature2 & Feature3 on each cluster of customers
# 
# Visualizing how key features are distributed within each customer cluster helps identify distinct characteristics and preferences of different segments. For example, one cluster may show a concentration of high-income individuals, while another might highlight younger customers.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# List of features to visualize
features = ["Feature1", "Feature2", "Feature3"]  

# Define a color palette for the visualizations
colors = ['#4E79A7', '#76B7B2', '#A0A0A0', '#9467BD']  # Vibrant professional colors

# Iterate over the clusters and plot distributions for each feature
clusters = clustered_df['Cluster'].unique()  # Extract unique cluster IDs

for cluster in clusters:
    cluster_data = clustered_df[clustered_df['Cluster'] == cluster]  # Filter data for the cluster
    plt.figure(figsize=(10, 6))  # Set figure size
    
    # Create a subplot for each feature
    for i, feature in enumerate(features, start=1):
        plt.subplot(1, len(features), i)
        
        # Plot the KDE plot for the feature
        sns.kdeplot(
            data=cluster_data, 
            x=feature, 
            fill=True, 
            color=colors[cluster], 
            alpha=0.6, 
            linewidth=1.5
        )
        
        # Add titles and labels
        plt.title(f'{feature} Distribution in Cluster {cluster}', fontsize=14, fontweight='bold')
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
    # Adjust layout for the cluster's feature visualizations
    plt.tight_layout()
    plt.suptitle(f'Distributions of Features for Cluster {cluster}', fontsize=16, fontweight='bold', y=1.02)
    
    # Show the plots for this cluster
    plt.show()


# %% [markdown]
# #### Insights from graphs                          
# ##### Cluster 0:
# Feature1:
# 
# Distribution: Concentrated in the lower range with low variability.
# Interpretation: This indicates a group with limited positive outcomes for Feature1. They are consistent but not high-performing in this metric.
# 
# Feature2:
# 
# Distribution: Skewed toward the lower-mid range, with minimal spread.
# Interpretation: Performance in Feature2 is moderate and stable, suggesting a group that responds well to predictable and cost-efficient strategies.
# 
# Feature3:
# 
# Distribution: Clustered tightly in the lower range, indicating limited variability.
# Interpretation: This group’s consistent but lower Feature3 values reflect a preference for standardized, affordable offerings.over-customization.
# 
# ##### Cluster 1:
# Feature1:
# 
# Distribution: Shows the highest values across clusters, with significant variability.
# Interpretation: Cluster 1 represents a dynamic group excelling in Feature1. They are open to personalized and high-value solutions.
# 
# Feature2:
# 
# Distribution: Centered in the mid-to-high range, with moderate variability.
# Interpretation: Reliable performance in Feature2 positions this group as a consistent but slightly varied segment, ideal for tiered service offerings.
# 
# Feature3:
# 
# Distribution: Similar to Feature1, with high values and variability.
# Interpretation: This group demonstrates strong, diverse outcomes, making them an ideal target for premium and flexible offerings that cater to varied preferences.
# 
# 
# 
# ##### Cluster 2:
# Feature1:
# 
# Distribution: Concentrated in the mid-range, with narrow spread.
# Interpretation: This cluster performs moderately and consistently in Feature1, indicating a steady but unremarkable group.
# 
# Feature2:
# 
# Distribution: Mid-range values with low variability.
# Interpretation: Consistent but average outcomes in Feature2 highlight their reliability and preference for stability, rather than innovation.
# 
# Feature3:
# 
# Distribution: Moderate values concentrated in the mid-range, with minimal variability.
# Interpretation: This group’s stable but less dynamic Feature3 performance makes them well-suited for scalable and standardized solutions.
# 
# ##### Cluster 3:
# Feature1:
# 
# Distribution: Skewed toward the upper mid-range, with moderate variability.
# Interpretation: This cluster achieves strong outcomes in Feature1, signaling their potential for growth through targeted engagement.
# 
# Feature2:
# 
# Distribution: Displays the highest values across all clusters, with significant variability.
# Interpretation: The dynamic nature of this cluster’s Feature2 outcomes suggests they are high-potential customers, ready for premium offerings or high-impact initiatives.
# 
# Feature3:
# 
# Distribution: Broadly spread across the mid-to-high range, with moderate variability.
# Interpretation: Cluster 3’s diverse yet strong Feature3 outcomes indicate an opportunity to adapt strategies to cater to varying customer needs.

# %% [markdown]
# #### Showing Total Number of Accepted Campaigns for each cluster
# 
# Analyzing the number of accepted marketing campaigns across clusters reveals which groups are most responsive to outreach efforts. This insight enables businesses to prioritize targeting strategies for more engaged customer segments.

# %%
# Optimized Visualization for Total Number of Accepted Campaigns per Cluster
plt.figure(figsize=(10, 6))  # Set figure size

# Create the countplot with Seaborn for cleaner aesthetics
pl = sns.countplot(
    x='No of Accepted Campains', 
    hue='Cluster', 
    data=df_org, 
    palette = ['#4E79A7', '#F28E2B', '#A0A0A0', '#9467BD']  
)

# Set the title and axis labels with consistent styling
pl.set_title("Total Number of Accepted Campaigns for Each Cluster", fontsize=16, fontweight='bold', pad=20)
pl.set_xlabel("Number of Accepted Campaigns", fontsize=14, labelpad=10)
pl.set_ylabel("Frequency (Count)", fontsize=14, labelpad=10)
pl.tick_params(axis='both', labelsize=12)

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate the bars with the count values
for p in pl.patches:
    height = p.get_height()  
    x_position = p.get_x() + p.get_width() / 2  
    # Add the text label
    pl.text(
        x_position, 
        height + 0.1,  
        f'{int(height)}', 
        ha="center", va="bottom", fontsize=10, color='black'
    )

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# ### Insights from the Graph:
# ##### **Cluster 3** has the highest number of individuals with "0 Accepted Campaigns," suggesting this group is the least responsive.
# ##### **Cluster 0** follows with a significant count in "0 Accepted Campaigns," also indicating low engagement.
# ##### **Cluster 2** has the highest representation in 2 and 3 accepted campaigns, suggesting it is the most engaged group.
# ##### **Cluster 1** has a balanced distribution but lower engagement, with fewer individuals in higher accepted campaign counts.
# ##### **Cluster 0** has a moderate number of individuals accepting exactly 1 campaign, indicating a small but notable level of engagement.

# %% [markdown]
# #### Plotting Family Sizes in each cluster
# 
# Visualizing family size distribution within clusters provides insights into household dynamics and consumption behaviors, which can inform product offerings or promotional campaigns tailored to families.
# 

# %%
# Optimized Visualization for Family Sizes in Each Cluster
plt.figure(figsize=(10, 6))  # Set a larger figure size for better readability

# Create the countplot with a professional color palette
pl = sns.countplot(
    x='Family_Size', 
    hue='Cluster', 
    data=df_org, 
    palette = ['#4E79A7', '#F28E2B', '#A0A0A0', '#9467BD']
)

# Set the title and axis labels with enhanced styling
pl.set_title("Family Sizes in Each Cluster", fontsize=16, fontweight='bold', pad=20)
pl.set_xlabel("Family Size", fontsize=14, labelpad=10)
pl.set_ylabel("Frequency (Count)", fontsize=14, labelpad=10)
pl.tick_params(axis='both', labelsize=12)

# Add gridlines for easier interpretation
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate the bars with the count values
for p in pl.patches:
    height = p.get_height()  # Get the bar height
    x_position = p.get_x() + p.get_width() / 2  # Calculate the center of the bar
    # Add the annotation with proper styling
    pl.text(
        x_position, 
        height + 0.1,  # Position above the bar
        f'{int(height)}',  # The count value
        ha="center", va="bottom", fontsize=10, color='black'
    )

# Adjust the legend styling for clarity
plt.legend(
    title='Cluster', 
    title_fontsize=12, 
    fontsize=10, 
    loc='upper right', 
    frameon=False
)

# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()


# %% [markdown]
# #### Insights from the graph:
# * **Cluster 0:**
#    - Leads in medium family sizes (3 members) with the highest frequency in this category.
#    - Has a strong presence in small families (2 members).
#    - **Strategy**: Offer bundles tailored to 2–3-member families, such as small household items or compact service packages.
# 
# * **Cluster 1:**
#    - Dominates in small family sizes (1 member) with the highest count.
#    - Moderate presence in 2-member families.
#    - **Strategy**: Target single individuals and small households with individual-focused offerings, such as single-serve products or compact services.
#    
# * **Cluster 2:** 
#    - Has a notable presence in small family sizes (1–2 members).
#    - Does not lead in any specific category but shows engagement in smaller households.
#    - **Strategy**: Target smaller families and individuals with personalized or portable solutions.
# 
# * **Cluster 3:** 
#    - Leads in large family sizes (4–5 members) and has the highest count of medium family sizes (3 members).
#    - **Strategy**: Tailor campaigns for medium and large families with offerings like family bundles, home appliances, or family-oriented services.

# %% [markdown]
# #### Using Seaborn (sns) to generate visualizations for the relationships between a set of personal features ('Income' , 'Family_Size', 'Education', ...etc.) and the amount spent by a customer.

# %% [markdown]
# Leveraging Seaborn’s visual tools, we can create plots to explore the relationship between a customer’s income and their spending patterns. These visualizations highlight trends, such as whether higher-income customers spend proportionally more, offering valuable guidance for pricing and targeting strategies.

# %%
from matplotlib.lines import Line2D

# Define a more vibrant color palette
palette = ['#003f5c', '#bc5090', '#ffa600', '#58508d']  # Deep Blue, Magenta, Bright Orange, Purple

# Create a KDE plot using Seaborn's jointplot
joint_plot = sns.jointplot(
    x='Income', 
    y='Total Amount Purchases', 
    hue='Cluster', 
    data=df_org, 
    kind="kde", 
    palette=palette, 
    fill=True,  
    alpha=0.7,  
    linewidth=1
)

# Add titles and labels with enhanced styling
joint_plot.fig.suptitle(
    'Relationships Between Income and Total Amount Purchases by Cluster', 
    fontsize=14, fontweight='bold', y=1.02
)
joint_plot.set_axis_labels(
    'Income ($)', 
    'Total Amount Purchases', 
    fontsize=14, 
    labelpad=10
)

# Manually add a legend for clarity
legend_elements = [
    Line2D([0], [0], color=palette[0], lw=4, label='Cluster 0 (Deep Blue)'),
    Line2D([0], [0], color=palette[1], lw=4, label='Cluster 1 (Magenta)'),
    Line2D([0], [0], color=palette[2], lw=4, label='Cluster 2 (Bright Orange)'),
    Line2D([0], [0], color=palette[3], lw=4, label='Cluster 3 (Purple)')
]

# Add the legend to the figure
#joint_plot.fig.legend(
#    handles=legend_elements, 
#    title='Cluster', 
#    title_fontsize=12, 
#    fontsize=10, 
#    loc='lower right', 
#    frameon=True
#)

# Adjust layout for better spacing
joint_plot.fig.subplots_adjust(top=0.93)

# Display the plot
plt.show()


# %% [markdown]
# ### **Insights from the Graph:**
# 
# **Cluster 0 (Deep Blue):**
# - **Income Range**: Concentrated in the lower-income range (approximately $20,000–$50,000).
# - **Total Purchases**: Primarily below 500, with very few higher spending individuals.
# - **Profile**: Represents a low-income, low-spending group.
# - **Strategy**: Engage Cluster 0 with budget-friendly strategies.
# 
# 
# **Cluster 1 (Magenta):**
# - **Income Range**: Centered around the middle-income range ($40,000–$80,000).
# - **Total Purchases**: Primarily between 500 and 1,500, indicating moderate spending behavior.
# - **Profile**: Represents a mid-income, moderate-spending group.
# - **Strategy**: Offer products or services that balance quality and affordability, with occasional premium upgrades.
# 
# 
# **Cluster 2 (Bright Orange):**
# - **Income Range**: Distributed in the upper-middle-income range ($60,000–$100,000).
# - **Total Purchases**: Ranges from 500 to over 2,000, indicating higher spending patterns.
# - **Profile**: Represents a high-income, high-spending group.
# - **Strategy**: Target Cluster 2 for premium offerings, such as luxury products, exclusive memberships, or high-end services.
# 
# 
# **Cluster 3 (Purple):**
# - **Income Range**: Spread across the middle-income range ($40,000–$90,000).
# - **Total Purchases**: Primarily between 500 and 1,500, indicating moderate to high spending.
# - **Profile**: Represents a mid-income, moderately high-spending group.
# - **Strategy**: Tailor mid-tier to high-value offerings, focusing on quality and family-oriented products or services.

# %% [markdown]
# #### Cluster Profiles and Suggested Marketing Strategies
# 
# ##### **Cluster 0 - Profile: Traditional Married Budget-Conscious Families
# 
# * Feature Insights:
# 
# Income: Concentrated in the low-income bracket, with minimal variability.
# Total Purchases: Focused on the low spending range, indicating strong price sensitivity.
# Interpretation: This group prioritizes basic necessities and cost-saving solutions for their families.
# 
# * Marketing Strategies:
# 
# Family Bundles: Create affordable combo packs like "Weekend Essentials" or "Budget Family Kits," including staples such as meat, fish, and candies.
# Community Engagement: Organize family-focused events like cooking workshops or children's activities where the store’s products are subtly integrated.
# Loyalty Programs: Introduce reward points or cashback offers for frequent purchases, encouraging long-term loyalty.
# Localized Offers: Collaborate with local stores or centers to create discounts tailored to community-specific needs.
# Promotions on Essentials: Provide heavy discounts on daily essentials (e.g., rice, flour, sugar) to attract cost-conscious shoppers.
# 
# ##### **Cluster 1 - Profile: Educated Singles & Widowed Professionals
# 
# * Feature Insights:
# 
# Income: Concentrated in the mid-income range, with moderate variability.
# Total Purchases: Spread across the mid-tier spending level, indicating a balance between quality and cost.
# Interpretation: Cluster 1 is composed of single professionals who value convenience, quality, and lifestyle-oriented products.
# 
# * Marketing Strategies:
# 
# Premium Ready-to-Eat Options: Promote pre-prepared meals, quick-cook fish, or snack-sized candy packs that cater to busy professionals.
# Customizable Experiences: Introduce personalized gift packs or premium offerings like high-quality chocolate assortments or imported wines.
# Digital Engagement: Create personalized recommendations and online promotions via an app or e-commerce platform.
# Upsell Premium Products: Offer limited-time deals on high-value items like artisanal goods or curated wine selections.
# Lifestyle Kits: Target professionals with curated kits, such as "Single Gourmet Pack" featuring healthy snacks, craft beers, or small portions of premium meats.
# 
# ##### **Cluster 2 - Profile: Educated Married High Earners
# 
# * Feature Insights:
# 
# Income: Concentrated in the high-income range, reflecting significant spending power.
# Total Purchases: Focused on the upper mid-tier, indicating consistent spending on high-quality products.
# Interpretation: This cluster represents families seeking premium, long-lasting products and convenience-driven solutions.
# 
# * Marketing Strategies:
# 
# Exclusive Membership Programs: Offer VIP memberships that provide benefits like free delivery, access to premium packaging, or concierge shopping services.
# Luxury Family Products: Promote premium quality goods such as gourmet fish, imported wines, or high-end chocolates tailored for family dining or celebrations.
# Health-Focused Items: Highlight organic or eco-friendly product options, such as sustainable seafood or grass-fed beef.
# Event Sponsorships: Partner with upscale events such as wine tastings or tech fairs to increase brand visibility and create positive associations.
# Subscription Services: Offer curated monthly subscription boxes featuring artisanal foods, premium groceries, or luxury snacks.
# 
# 
# ##### **Cluster 3 - Profile: Urban Educated Singles
# 
# * Feature Insights:
# 
# Income: Spread across the middle-to-upper income range, with moderate variability.
# Total Purchases: Covers a wide range, from moderate to high spending, reflecting their dynamic and adaptable nature.
# Interpretation: Cluster 3 is composed of tech-savvy, experience-driven individuals who appreciate unique and innovative products.
# 
# * Marketing Strategies:
# 
# Exclusive Experiences: Offer products like "International Candy Samplers" or "Gourmet Tasting Boxes" to satisfy their curiosity for new experiences.
# Digital Promotions: Engage through influencer campaigns, flash sales, or early-access deals for trendy or unique products.
# Eco-Friendly Options: Emphasize sustainable or reusable products, such as eco-friendly packaging for meat or plant-based snacks.
# Premium Tech-Integrated Products: Promote high-tech kitchen gadgets, sleek packaging, or innovative food items that appeal to their modern lifestyle.
# Event-Based Offers: Organize tech-themed promotions, such as virtual wine-tasting events paired with premium snacks.
# 
# ##### **Overall Strategies
# 
# * Segmentation-Driven Offerings:
# 
# Tailor products for each cluster: budget-friendly for Cluster 0, convenience-oriented for Cluster 1, premium and exclusive for Cluster 2, and trendy or experiential for Cluster 3.
# 
# * Unified Loyalty Program:
# 
# Develop tiered loyalty systems where Cluster 0 earns points for value purchases, while Clusters 2 and 3 enjoy perks like VIP services or exclusive events.
# 
# * Seasonal and Festive Campaigns:
# 
# Launch holiday-specific bundles tailored to each cluster. For instance, luxury holiday gift boxes for Cluster 2 and unique seasonal flavors for Cluster 3.
# 
# * Hybrid Marketing Channels:
# 
# Use traditional approaches for Cluster 0, such as in-store displays and local promotions, while leveraging social media and digital platforms for Clusters 1, 2, and 3.
# 
# * Collaborative Partnerships:
# 
# Partner with family-friendly brands for Cluster 0, professional networks for Cluster 1, luxury providers for Cluster 2, and eco-friendly or tech brands for Cluster 3.
# 


