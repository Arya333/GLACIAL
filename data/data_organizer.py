import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv('data/PPMI_Curated_Data_Cut_Public_20230612_rev.csv')

# Filter DataFrame to have patients with atleast min_records visits recorded
min_records = 4
filtered_df = df.groupby('PATNO').filter(lambda x: len(x) >= min_records)

#print(filtered_df.nunique())

# Sort the DataFrame by 'PATNO' in ascending order
sorted_df = filtered_df.sort_values(by='PATNO', ascending=True)

columns = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs3_score_on", "updrs4_score", "updrs_totscore", "updrs_totscore_on"]
columns.extend(["age_at_visit", "PATNO", "YEAR"])

result_df = sorted_df[columns]
result_df.rename(columns={'age_at_visit' : 'AGE', 'PATNO': 'RID', 'YEAR': 'Year'}, inplace=True)

print(result_df.head())

result_df.to_csv('data/data1.csv', index=False)

