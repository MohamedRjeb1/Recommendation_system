import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parfums_carac=pd.read_csv('parfums_300.csv')
avis_user=pd.read_csv('avis_300.csv')
parfums_carac.iloc[25:40 , :]
avis_user.head()
# parfums_carac.info()
# avis_user.info()



parfums_carac.set_index('id',inplace=True)
avis_user.set_index('id_utilisateur',inplace=True)
print(parfums_carac.head())
print(avis_user.head())




merged_df = pd.merge(avis_user, parfums_carac, left_on='id_parfum', right_index=True, how='inner')

# Group by brand name and calculate the average rating for each brand
brand_ratings = merged_df.groupby('marque')['note'].mean()
brand_ratings.sort_values(ascending=False,inplace=True)


# Create the histogram
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
brand_ratings.plot(kind='bar')
plt.xlabel("Brand Name")
plt.ylabel("Average Rating")
plt.title("Average Rating per Brand")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()




# Group by brand and count the number of users for each brand
brand_user_counts = merged_df.groupby('marque').size()
brand_user_counts.sort_values(ascending=False,inplace=True)



# Create the bar plot
plt.figure(figsize=(10, 6))
brand_user_counts.plot(kind='bar',)
plt.xlabel("Brand Name")
plt.ylabel("Number of Users")
plt.title("Number of Users per Brand")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()