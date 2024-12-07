# Customer Personality Analysis Project

## Project Overview
Welcome to the **Customer Personality Analysis Project!** This project uses a dataset of customer information to optimal customer clusters using robust clustering methods, and subsequently tailor marketing strategies for each cluster to drive higher revenue and customer engagement.
Using various machine learning techniques. Our team has analyzed, cleansed and manipulated the data by adding more information based the existing data like totals, Age, …etc., and done a dimensional reduction to prepare it for clustering using KMeans and elbow methods.
After that, profiling of clusters done by the team to better understand the clusters so that we can derive actionable recommendations for the marketing team.

## Team Members

- **Bashar Nusir** ([bnusir](https://github.com/bnusir))
- **Carlos fuentes** ([kfuentes0524](https://github.com/kfuentes0524))
- **Minsang Kim** ([MinsangKim-Data](https://github.com/MinsangKim-Data))
- **Priyank Srivastava** ([Priyank315](https://github.com/Priyank315))
- **Yibin Wang** ([funny-carrot](https://github.com/funny-carrot))

## Objectives
*   Cluster Analysis:
    -   Determine the optimal number of customer groups.
    -   Analyze demographic and behavioral data to segment customers effectively.
*   Marketing Insights:
    -   Generate actionable recommendations for each identified customer group to improve campaign targeting and effectiveness.

## Dataset Details
*   Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
* 	Description:
    -   Contains customer demographic attributes
    -   Contains customer behavioral attributes
    -   Used to explore patterns to optimize marketing strategies 
    -	Format: Tab-Separated Values (TSV)

## Assumptions
1.	Data Consistency:
    -   Assumed  “Graduation” equals “Bachelor” and “2n Cycle” equals “Master” 
    - 	Assumed  “Together” equals “Married” and “Alone” equals “Single” 
2.	Outlier Removal: 
    -   Income greater than 125K and ages over 100 years were treated as outliers.

## Data Cleansing  
1.  Dropping observations with meaningless values as: 
    -  'Absurd' and 'YOLO' values under marital status 
2.  Dropping columns with meaningless values as: 
    -   Z_CostContact   =3 always 
    -   Z_Revenue =11 always
  
###  Dataset Adaptation: 
*   Original TSV file renamed and reformatted for easier handing. 

## Tools and Frameworks
### Libraries:
* Pandas, NumPy for data manipulation.
* Seaborn, Matplotlib for visualizations.
* Scikit-learn for clustering
* KElbowVisualizer for determining optimal number of clusters
### Python Version 3.12.3

## Project Outputs
### Cluster Profiles:
*   Customers were segmented based on: 
    -    Customer demographics 
    -    Customer behavioral traits 
### Marketing Recommendations

## Rules of Engagement
1.	Transparent and clear communication among team members.
2.	Collaboration and teamwork for effective results.
3.	Regular updates and reviews to ensure progress alignment.

## Visualizations

<p>
  <img src="reports/Customers_Age_range.png" alt="Image 1" height="200" width="300" style="display:inline;">
  <img src="reports/Customers_Education_level.png" alt="Image 2" height="200" width="300" style="display:inline;">
  <img src="reports/Customers_Income.png" alt="Image 3" height="200" width="300" style="display:inline;">
  <img src="reports/Customers_Income_range.png" alt="Image 4" height="200" width="300" style="display:inline;">
  <img src="reports/Customers_Marital_status.png" alt="Image 5" height="200" width="300" style="display:inline;">
  <img src="reports/Number_of_Purchases_by_Age.png" alt="Image 6" height="200" width="300" style="display:inline;">
</p>

**Analysis of Customer Data**

- **Objective:** To understand key factors related to customer demographics, such as marital status, education level, income level, and total purchases.

- **Visualizations:**
  - **Bar Chart Plots:** To understand customers demographics  and behavioral traits.

#### **Audience:**

- Marketing team and Data Scientists.

<br>

---
### **Part 2 of the Project: Segmenting Customers into groups**

<img src="reports/3D_Cluster_Visualization_of_PCA_Components.png" alt="Image" height="200" width="300" >

#### **Segmentation or clustering customers in groups:**  
Grouping customers based on similar traits and shared characteristics such as income, age, spending, …etc., and since we did not know what the best combinations are, we used unsupervised method to analyze the data and cluster customers in groups.

- **Objective:** To visualize the clusters or groups of the customer. The goal is to group customers with shared characteristics and traits.
- **Insights:** By analyzing the dominant features within each cluster and each feature contribution in each cluster, we gain insights into the key characteristics that define each group:
    -    Feature 1's top 3 influential variables are 'Income', 'NumCatalogPurchases', and 'MntMeatProducts'
    -    Feature 2's top 3 influential variables are 'Teenhome', 'NumDealsPurchases', and 'NumWebPurchases'
    -    Feature 3's top 3 influential variables are 'AcceptedCmp4', 'Response', 'AcceptedCmp2'

#### **Audience:**

- Marketing team and Data Scientists.

#### **Visualization:**

<p>
  <img src="reports/Distortion_Score_Elbow_For_KMeans_Clustering.png" alt="Image 1" height="200" width="300" style="display:inline;">
  <img src="reports/PCA_Dimentionality_Reduction_3D.png" alt="Image 1" height="200" width="300" style="display:inline;">
  <img src="reports/Principal_Components_Domenant_Original_Values.png" alt="Image 5" height="200" width="300" style="display:inline;">  
  <img src="reports/Distribution_of_Features_in_Cluster_0.png" alt="Image 3" height="200" width="300" style="display:inline;">
  <img src="reports/Distribution_of_Features_in_Cluster_1.png" alt="Image 4" height="200" width="300" style="display:inline;">
  <img src="reports/Distribution_of_Features_in_Cluster_2.png" alt="Image 5" height="200" width="300" style="display:inline;">
  <img src="reports/Distribution_of_Features_in_Cluster_3.png" alt="Image 6" height="200" width="300" style="display:inline;">
  <img src="reports/Percentage_Distribution_by_Category_and_Cluster.png" alt="Image 7" height="200" width="300" style="display:inline;">
  <img src="reports/3D_Cluster_Visualization_of_PCA_Components.png" alt="Image 8" height="200" width="300" style="display:inline;">
 </p>

- **Line plot:** Using Elbow method to determine the optimal number of customer groups/clusters.
- **3D plot:** To Visualize the clusters(groups) of customers.
- **Correlation Heatmap Analysis plot** A powerful visualization tool to show relationships between different variables and their influence on each feature
- **Density plot:** To Visualize the Density of each feature in each clusters of customers.

<br>

---

### **Part 3 of the Project: Analysis of Customer groups**

#### **Detailed Interpretation of Relationship between clusters and original features (Income, Marital Status, Education, ...etc.)**

  Developed using Python with Seaborn by matplotlib, this Seaborn offers easy to plot rich graphs with information and relationships.

<img src="reports/Number_of_Customers_in_Each_Cluster.png" alt="Image" height="200" width="300">

- **Number of customers in each cluster**  
  - **Observations:** \
      The number of data points in each cluster varies significantly:
       - Cluster 3 has the largest size with 964 data points.
       - Cluster 2 is the smallest with 145 data points.
       - Clusters 1 and 0 have 476 and 587 data points, respectively.
  - **Insights:**-
       - Cluster 3: Likely represents a broad group of customers with common or average characteristics. characteristics.
       - Cluster 2: Represents a niche segment, possibly an outlier group or a specialized demographic.
       - Cluster sizes suggest different levels of customer importance or distinctiveness.

<img src="reports/Total_No_of_Accepted_Campains_for_Each_Cluster.png" alt="Image" height="200" width="300">

- **Total Number of Accepted Campaigns for Each Cluster**  
  - **Observations & Insights:** \
        - Cluster 3 has the highest number of individuals with "0 Accepted Campaigns", suggesting this group is the least responsive. \
        - Cluster 0 follows with a significant count in "0 Accepted Campaigns", also indicating low engagement. \
        - Cluster 2 has the highest representation in 2 and 3 accepted campaigns, suggesting it is the most engaged group. \
        - Cluster 1 has a balanced distribution but lower engagement, with fewer individuals in higher accepted campaign counts. \
        - Cluster 0 has a moderate number of individuals accepting exactly 1 campaign, indicating a small but notable level of engagement.


<img src="reports/Family_Sizes_in_each_Cluster.png" alt="Image" height="200" width="300">

- **Family Sizes in Each Cluster**  
  - **Observations & Insights:** \
        -     Cluster 0 leads in meduim family sizes (3 members) and small family size of (1 to 2) members (Small to Meduim Families)\
                     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-   Consider offering bundles tailored to 2–3-member families. \
        -     Cluster 1 most in small family size of (1 to 2) members (Small Families) \
                  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;-     Target single individuals with individual-focused offerings, such as single-serve products \
        -     Cluster 2 leads in small family size of (1 to 2) members (Small Families) \
                  &nbsp;&nbsp;  &nbsp;&nbsp; &nbsp;&nbsp;-     Target single individuals with individual-focused offerings, such as single-serve products \
        -     Cluster 3 leads in large family sizes (4–5 members) and meduim family size of 3 members (Meduim to Larg Families) \
                   &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;-     Tailor campaigns for medium and large families, such as family bundles, home appliances, or family-oriented services.

<img src="reports/Relationships_Income_and_Total_Amount_Purchases_by_Cluster.png" alt="Image" height="200" width="300">

- **Relationships Between Income and Total Amount Purchases by Cluster**  
- &nbsp;&nbsp;**Observations & Insights:** 
  
  &nbsp;&nbsp; - **Cluster 0 (Deep Blue):**
    - &nbsp;&nbsp;&nbsp;&nbsp;**Income Range**: Concentrated in the lower-income range (approximately $20,000–$50,000).
    - &nbsp;&nbsp;&nbsp;&nbsp;**Total Purchases**: Primarily below 500, with very few higher spending individuals.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Profile**: Represents a low-income, low-spending group.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Strategy**: Engage Cluster 0 with budget-friendly strategies.

  &nbsp;&nbsp; - **Cluster 1 (Magenta):** 
    - &nbsp;&nbsp;&nbsp;&nbsp;**Income Range**: Centered around the middle-income range ($40,000–$80,000).
    - &nbsp;&nbsp;&nbsp;&nbsp;**Total Purchases**: Primarily between 500 and 1,500, indicating moderate spending behavior.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Profile**: Represents a mid-income, moderate-spending group. 
    - &nbsp;&nbsp;&nbsp;&nbsp; **Strategy**: Offer products or services that balance quality and affordability, with occasional premium upgrades. 
    
  &nbsp;&nbsp; - **Cluster 2 (Bright Orange):**
    - &nbsp;&nbsp;&nbsp;&nbsp;**Income Range**: Distributed in the upper-middle-income range ($60,000–$100,000).
    - &nbsp;&nbsp;&nbsp;&nbsp;**Total Purchases**: Ranges from 500 to over 2,000, indicating higher spending patterns.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Profile**: Represents a high-income, high-spending group.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Strategy**: Target Cluster 2 for premium offerings, such as luxury products, exclusive memberships, or high-end services.

  &nbsp;&nbsp; - **Cluster 3 (Purple):**
    - &nbsp;&nbsp;&nbsp;&nbsp;**Income Range**: Spread across the middle-income range ($40,000–$90,000).
    - &nbsp;&nbsp;&nbsp;&nbsp;**Total Purchases**: Primarily between 500 and 1,500, indicating moderate to high spending.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Profile**: Represents a mid-income, moderately high-spending group.
    - &nbsp;&nbsp;&nbsp;&nbsp;**Strategy**: Tailor mid-tier to high-value offerings, focusing on quality and family-oriented products or services. 

#### Cluster Profiles and Suggested Marketing Strategies

##### **Cluster 0 - Profile: Traditional Married Budget-Conscious Families

* Feature Insights:

Income: Concentrated in the low-income bracket, with minimal variability.
Total Purchases: Focused on the low spending range, indicating strong price sensitivity.
Interpretation: This group prioritizes basic necessities and cost-saving solutions for their families.

* Marketing Strategies:

Family Bundles: Create affordable combo packs like "Weekend Essentials" or "Budget Family Kits," including staples such as meat, fish, and candies.
Community Engagement: Organize family-focused events like cooking workshops or children's activities where the store’s products are subtly integrated.
Loyalty Programs: Introduce reward points or cashback offers for frequent purchases, encouraging long-term loyalty.
Localized Offers: Collaborate with local stores or centers to create discounts tailored to community-specific needs.
Promotions on Essentials: Provide heavy discounts on daily essentials (e.g., rice, flour, sugar) to attract cost-conscious shoppers.

##### **Cluster 1 - Profile: Educated Singles & Widowed Professionals

* Feature Insights:

Income: Concentrated in the mid-income range, with moderate variability.
Total Purchases: Spread across the mid-tier spending level, indicating a balance between quality and cost.
Interpretation: Cluster 1 is composed of single professionals who value convenience, quality, and lifestyle-oriented products.

* Marketing Strategies:

Premium Ready-to-Eat Options: Promote pre-prepared meals, quick-cook fish, or snack-sized candy packs that cater to busy professionals.
Customizable Experiences: Introduce personalized gift packs or premium offerings like high-quality chocolate assortments or imported wines.
Digital Engagement: Create personalized recommendations and online promotions via an app or e-commerce platform.
Upsell Premium Products: Offer limited-time deals on high-value items like artisanal goods or curated wine selections.
Lifestyle Kits: Target professionals with curated kits, such as "Single Gourmet Pack" featuring healthy snacks, craft beers, or small portions of premium meats.

##### **Cluster 2 - Profile: Educated Married High Earners

* Feature Insights:

Income: Concentrated in the high-income range, reflecting significant spending power.
Total Purchases: Focused on the upper mid-tier, indicating consistent spending on high-quality products.
Interpretation: This cluster represents families seeking premium, long-lasting products and convenience-driven solutions.

* Marketing Strategies:

Exclusive Membership Programs: Offer VIP memberships that provide benefits like free delivery, access to premium packaging, or concierge shopping services.
Luxury Family Products: Promote premium quality goods such as gourmet fish, imported wines, or high-end chocolates tailored for family dining or celebrations.
Health-Focused Items: Highlight organic or eco-friendly product options, such as sustainable seafood or grass-fed beef.
Event Sponsorships: Partner with upscale events such as wine tastings or tech fairs to increase brand visibility and create positive associations.
Subscription Services: Offer curated monthly subscription boxes featuring artisanal foods, premium groceries, or luxury snacks.


##### **Cluster 3 - Profile: Urban Educated Singles

* Feature Insights:

Income: Spread across the middle-to-upper income range, with moderate variability.
Total Purchases: Covers a wide range, from moderate to high spending, reflecting their dynamic and adaptable nature.
Interpretation: Cluster 3 is composed of tech-savvy, experience-driven individuals who appreciate unique and innovative products.

* Marketing Strategies:

Exclusive Experiences: Offer products like "International Candy Samplers" or "Gourmet Tasting Boxes" to satisfy their curiosity for new experiences.
Digital Promotions: Engage through influencer campaigns, flash sales, or early-access deals for trendy or unique products.
Eco-Friendly Options: Emphasize sustainable or reusable products, such as eco-friendly packaging for meat or plant-based snacks.
Premium Tech-Integrated Products: Promote high-tech kitchen gadgets, sleek packaging, or innovative food items that appeal to their modern lifestyle.
Event-Based Offers: Organize tech-themed promotions, such as virtual wine-tasting events paired with premium snacks.

##### **Overall Strategies

* Segmentation-Driven Offerings:

Tailor products for each cluster: budget-friendly for Cluster 0, convenience-oriented for Cluster 1, premium and exclusive for Cluster 2, and trendy or experiential for Cluster 3.

* Unified Loyalty Program:

Develop tiered loyalty systems where Cluster 0 earns points for value purchases, while Clusters 2 and 3 enjoy perks like VIP services or exclusive events.

* Seasonal and Festive Campaigns:

Launch holiday-specific bundles tailored to each cluster. For instance, luxury holiday gift boxes for Cluster 2 and unique seasonal flavors for Cluster 3.

* Hybrid Marketing Channels:

Use traditional approaches for Cluster 0, such as in-store displays and local promotions, while leveraging social media and digital platforms for Clusters 1, 2, and 3.

* Collaborative Partnerships:

Partner with family-friendly brands for Cluster 0, professional networks for Cluster 1, luxury providers for Cluster 2, and eco-friendly or tech brands for Cluster 3.
## Video Links

- [Bashar Nusir]()
- [Carlos fuentes] (https://vimeo.com/1037047949/b6adc5e9ca?ts=0&share=copy)
- [Minsang Kim] (https://www.loom.com/share/a4aec0e1064d49dfac74b83d936a2fda)
- [Priyank Srivastava] ()
- [Yibin Wang]

## Sources and references utilized for this project 
  - [Data folder](./data/)
  - [Source Code](./src/)
  
 
