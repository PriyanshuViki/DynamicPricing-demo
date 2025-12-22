import pandas as pd
import os



# Script to remove discontinued materials from CSV files


# csv_dir = 'data/raw/CSV'
# output_dir = 'data/processed'
# os.makedirs(output_dir, exist_ok=True)

# material_columns = ['Material', 'MATNR', 'Material Number']

# for file in os.listdir(csv_dir):
#     if file.endswith('.csv'):
#         df = pd.read_csv(os.path.join(csv_dir, file))
        
#         for col in material_columns:
#             if col in df.columns:
#                 original = len(df)
#                 df = df[~df[col].astype(str).str.startswith('NV')]
#                 removed = original - len(df)
                
#                 if removed > 0:
#                     print(f"{file}: Removed {removed} discontinued materials")
#                 break
        
#         df.to_csv(os.path.join(output_dir, file), index=False)

# print("Processing complete!")

####################################################################################

# Script to print column Names

# csv_dir = 'data/raw/CSV'

# for file in os.listdir(csv_dir):
#     if file.endswith('.csv'):
#         df = pd.read_csv(os.path.join(csv_dir, file))
#         print(f"\n{file}:")
#         print(df.columns.tolist())


###########################################################################


#Script to read XLSX and save as CSV

mvke = pd.read_excel('data/raw/MVKE.XLSX')

print("MVKE columns:")
print(mvke.columns.tolist())
print(f"\nShape: {mvke.shape}")
print(f"\nFirst few rows:")
print(mvke.head())

# Save to CSV
os.makedirs('data/processed', exist_ok=True)
mvke.to_csv('data/processed/MVKE.csv', index=False)
print(f"\n✓ MVKE.csv saved to data/processed/")


###########################################################################


# Check data availability across all cost columns
# mbew = pd.read_csv('data/processed/MBEW.csv', low_memory=False)

# # Extract materials with pricing data for dynamic pricing model
# pricing_data = mbew[['Material', 'Valuation area', 'Moving price', 'MovAvgPrice PP', 
#                      'MovAvgPrice PY', 'Standard price', 'Price unit', 'Total Stock', 
#                      'Total Value', 'Price control']].copy()

# # Filter materials with current moving price
# pricing_data = pricing_data[pricing_data['Moving price'] > 0]

# # Calculate price trends
# pricing_data['Price_Change_PP'] = ((pricing_data['Moving price'] - pricing_data['MovAvgPrice PP']) 
#                                     / pricing_data['MovAvgPrice PP'] * 100)
# pricing_data['Price_Change_PY'] = ((pricing_data['Moving price'] - pricing_data['MovAvgPrice PY']) 
#                                     / pricing_data['MovAvgPrice PY'] * 100)

# print(f"Materials with pricing data: {len(pricing_data)}")
# print(f"\nPrice trends:")
# print(pricing_data[['Material', 'Moving price', 'Price_Change_PP', 'Price_Change_PY']].head(10))

# # Save for modeling
# pricing_data.to_csv('data/processed/pricing_features.csv', index=False)
# print(f"\n✓ Saved to pricing_features.csv")


