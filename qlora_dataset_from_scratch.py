import pandas as pd
import json

# Load the Excel file
file_path = 'path_to_your_excel_file.xlsx'
output_json_path = 'path_to_save_json/qlora_formatted_data.json'

# Load the specific sheet
sheet_name = '(SF) Payments'
sf_payments = pd.read_excel(file_path, sheet_name=sheet_name)

# Filter relevant columns for the task
relevant_columns = ['Field Name (SF)', 'Field Name (SAP)']
sf_to_sap_mapping = sf_payments[relevant_columns].dropna()

# Format the data for QLoRA training
qlora_formatted_data = []

for _, row in sf_to_sap_mapping.iterrows():
    # Extract Salesforce and SAP fields
    sf_fields = sf_to_sap_mapping['Field Name (SF)'].tolist()
    sap_field = row['Field Name (SAP)']

    # Construct instruction, input, and response
    instruction = "Your task is to find the SAP value based on the input JSON."
    input_json = {field: "value" for field in sf_fields if field != sap_field}
    response = f"SAP-value = {sap_field}"

    # Append to the formatted data list
    qlora_formatted_data.append({
        "Instruction": instruction,
        "Input": input_json,
        "Response": response
    })

# Save the formatted data as a JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(qlora_formatted_data, json_file, indent=4)

print(f"Formatted data has been saved to {output_json_path}")
