import pandas as pd

filePath = "opa_properties_public.csv"

# Read the dataset
all_data = pd.read_csv(filePath, low_memory=False)

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------------------------
missingValueSummaryOutputFile = "missing_values_summary.txt"
highMissingValuesOutputFile = "columns_with_high_missing_values.txt"

# Calculate missing values for all columns
missing_values = all_data.isnull().sum()

# Save missing values summary for all columns to a text file
with open(missingValueSummaryOutputFile, 'w') as f:
    print("Missing values summary for all columns:", file=f)
    print(missing_values, file=f)

print("Missing values summary for all columns has been saved to", missingValueSummaryOutputFile)

# Calculate the threshold for identifying columns with more than 80% empty space
threshold = 0.8 * len(all_data)

# Filter columns with more than 80% empty space
columns_with_high_missing_values = missing_values[missing_values > threshold]

# Save columns with more than 80% empty space to a text file
with open(highMissingValuesOutputFile, 'w') as f:
    print("Columns with more than 80% empty space:", file=f)
    print(columns_with_high_missing_values, file=f)

print("Columns with more than 80% empty space have been saved to", highMissingValuesOutputFile)
print(all_data.columns)

# ---------------------------------------------------------------------------------------------
# Mark columns that has more than 80% missing values
# Mark columns that aren't going to make much impact to reduce the size of data
unnecessaryColumns = ['the_geom', 'the_geom_webmercator', 'assessment_date', 'beginning_point', 'book_and_page',
                      'building_code', 'category_code_description', 'census_tract', 'cross_reference',
                      'date_exterior_condition', 'fuel', 'garage_type', 'geographic_ward', 'house_extension',
                      'house_number', 'location', 'mailing_address_1', 'mailing_address_2', 'mailing_care_of',
                      'mailing_city_state', 'mailing_street', 'mailing_zip', 'market_value_date', 'number_of_rooms',
                      'other_building', 'owner_1', 'owner_2', 'parcel_number', 'recording_date', 'registry_number',
                      'sale_date', 'separate_utilities', 'sewer', 'site_type', 'state_code', 'street_designation',
                      'street_direction', 'street_name', 'suffix', 'topography', 'unfinished', 'unit', 'utility',
                      'year_built_estimate', 'objectid']

data = all_data.drop(unnecessaryColumns, axis=1)

print(data.columns) 
print(data.shape)
