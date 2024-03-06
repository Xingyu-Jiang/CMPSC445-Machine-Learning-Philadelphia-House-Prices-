import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()

filePath = "opa_properties_public.csv"

# Read the dataset
all_data = pd.read_csv(filePath, low_memory=False)

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ---------------------------------------------------------------------------------------------

# Calculate missing values for all columns
missing_values = all_data.isnull().sum()

# Calculate the threshold for identifying columns with more than 80% empty space
threshold = 0.8 * len(all_data)

# Filter columns with more than 80% empty space
columns_with_high_missing_values = missing_values[missing_values > threshold]

# Get the names of columns with high missing values into a list
high_missing_columns_list = columns_with_high_missing_values.index.tolist()
# print(high_missing_columns_list)

# ---------------------------------------------------------------------------------------------
# Mark columns that has more than 80% missing values
# Mark columns that aren't going to make much impact to reduce the size of data

unnecessaryColumns = ['assessment_date', 'beginning_point', 'book_and_page', 'building_code',
                      'building_code_description', 'building_code_description_new', 'building_code_new',
                      'category_code_description', 'census_tract', 'geographic_ward', 'house_number', 'location',
                      'mailing_address_1', 'mailing_city_state', 'mailing_street', 'mailing_zip', 'objectid', 'owner_1',
                      'owner_2', 'parcel_number', 'pin', 'recording_date', 'registry_number', 'sale_date', 'sale_price',
                      'state_code', 'street_designation', 'street_direction', 'street_name', 'the_geom',
                      'the_geom_webmercator', 'topography', 'year_built_estimate']

columnsToDrop = unnecessaryColumns + high_missing_columns_list

# There are sale price and market value two value that can be used to represet house price.
# Let use market value.

data = all_data.drop(columnsToDrop, axis=1)

# Drop rows that do not have a market value or equal to 0
data = data.dropna(subset=['market_value'])
data = data.drop(data[data.market_value == 0].index)

# ---------------------------------------------------------------------------------------------
# Convert categorical variables and Handling Missing Values
# Category code
values_to_remove = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
data = data[~data['category_code'].isin(values_to_remove)]

# # Perform one-hot encoding for the 'category_code' column inplace
# data = pd.get_dummies(data, columns=['category_code'], prefix='category')

encoded_data = label_encoder.fit_transform(data['category_code'])
data['category_code'] = encoded_data

# lat and lng
data["lat"] = data.lng.fillna(data.lng.mean())
data.loc[:, "lat"] = data['lat'].abs()

data["lng"] = data.lng.fillna(data.lng.mean())
data.loc[:, "lng"] = data['lng'].abs()

# Zipcode
data.dropna(subset=['zip_code'], inplace=True)

# Basement
data.loc[:, "basements"] = data.loc[:, "basements"].fillna('L')
data.loc[:, "basements"] = data['basements'].replace('0', 'K')
data.loc[:, "basements"] = data['basements'].replace('1', 'M')
data.loc[:, "basements"] = data['basements'].replace('2', 'N')
data.loc[:, "basements"] = data['basements'].replace('3', 'O')
data.loc[:, "basements"] = data['basements'].replace('4', 'P')
encoded_data = label_encoder.fit_transform(data['basements'])
data['basements'] = encoded_data

# Central air
data.loc[:, "central_air"] = data['central_air'].replace('0', 'N')
data.loc[:, "central_air"] = data['central_air'].replace('1', 'N')
data.loc[:, "central_air"] = data['central_air'].fillna('N')
# data['central_air'] = data['central_air'].replace({'N': False, 'Y': True})
encoded_data = label_encoder.fit_transform(data['central_air'])
data['central_air'] = encoded_data

# Depth
data.loc[:, "depth"] = data['depth'].fillna(0)

# Exterior condition
data.loc[:, "exterior_condition"] = data['exterior_condition'].fillna(0)
encoded_data = label_encoder.fit_transform(data['exterior_condition'])
data['exterior_condition'] = encoded_data

# Fireplaces
data.loc[:, "fireplaces"] = data['fireplaces'].fillna(0)
encoded_data = label_encoder.fit_transform(data['fireplaces'])
data['fireplaces'] = encoded_data

# Frontage
data.loc[:, "frontage"] = data['frontage'].fillna(0)

# Garage spaces
data.loc[:, "garage_spaces"] = data['garage_spaces'].fillna(0)

# General Construction
data.loc[:, "general_construction"] = data['general_construction'].fillna('G')
encoded_data = label_encoder.fit_transform(data['general_construction'])
data['general_construction'] = encoded_data

# Interior Condition
data.loc[:, "interior_condition"] = data['interior_condition'].fillna(0)
encoded_data = label_encoder.fit_transform(data['interior_condition'])
data['interior_condition'] = encoded_data

# Number of Bathrooms
data.loc[:, "number_of_bathrooms"] = data['number_of_bathrooms'].fillna(0)

# Number of Bedrooms
data.loc[:, "number_of_bedrooms"] = data['number_of_bedrooms'].fillna(0)

# Number stories
data.loc[:, "number_stories"] = data['number_stories'].fillna(0)

# Off Street Open
median_off_street_open = data['off_street_open'].median()
data.loc[:, "off_street_open"] = data['off_street_open'].fillna(median_off_street_open)

# Parcel shape
data.loc[:, "parcel_shape"] = data['parcel_shape'].fillna('F')
data.loc[:, "parcel_shape"] = data['parcel_shape'].replace(' ', 'F')
encoded_data = label_encoder.fit_transform(data['parcel_shape'])
data['parcel_shape'] = encoded_data

# Quality Grade
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'D-': 1, 'D ': 1, 'D+': 1, '1 ': 1})
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'E-': 2, 'E ': 2, 'E+': 2, '2 ': 2})
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'C-': 3, 'C ': 3, 'C+': 3, '3 ': 3})
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'B-': 4, 'B ': 4, 'B+': 4, '4 ': 4})
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'A-': 5, 'A ': 5, 'A+': 5, '5 ': 5})
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'S-': 6, 'S ': 6, 'S+': 6, '6 ': 6})
data.loc[:, "quality_grade"] = data['quality_grade'].replace({'X-': 7, 'X ': 7, 'X+': 7, '7 ': 7})
data.drop(data[data['quality_grade'] == '0 '].index, inplace=True)
median_quality_grade = data['quality_grade'].median()
data.loc[:, "quality_grade"] = data['quality_grade'].fillna(median_quality_grade).astype(int)
encoded_data = label_encoder.fit_transform(data['quality_grade'])
data['quality_grade'] = encoded_data

# Street Code
data.dropna(subset=['street_code'], inplace=True)
encoded_data = label_encoder.fit_transform(data['street_code'])
data['street_code'] = encoded_data

# Taxable Building
data.dropna(subset=['taxable_building'], inplace=True)

# Total Area
median_total_area = data['total_area'].median()
data.loc[:, "total_area"] = data['total_area'].fillna(median_total_area).astype(int)

# Total livable area
median_total_area = data['total_livable_area'].median()
data.loc[:, "total_livable_area"] = data['total_livable_area'].fillna(median_total_area).astype(int)

# Type Heater
data.loc[:, "type_heater"] = data['type_heater'].replace('0', 'H')
data.loc[:, "type_heater"] = data['type_heater'].fillna('H')
encoded_data = label_encoder.fit_transform(data['type_heater'])
data['type_heater'] = encoded_data

# View type
data.loc[:, "view_type"] = data['view_type'].replace('0', 'N ')
data.loc[:, "view_type"] = data['view_type'].fillna('N ')
encoded_data = label_encoder.fit_transform(data['view_type'])
data['view_type'] = encoded_data

# Year Built
median_year_built = data['year_built'].median()
data.loc[:, "year_built"] = data['year_built'].fillna(median_year_built).astype(int)

# Zoning
data.dropna(subset=['zoning'], inplace=True)
encoded_data = label_encoder.fit_transform(data['zoning'])
data['zoning'] = encoded_data

# ---------------------------------------------------------------------------------------------

# Define the file path for the new CSV file
processedDataFilePath = "processed_data.csv"

# Save the new data to a CSV file
data.to_csv(processedDataFilePath, index=False)

print("New data has been saved to", processedDataFilePath)

# print(data.shape)
# print(data.columns)

# Show and calculate missing values for all columns
print("Columns and amount of missing values")
missing_values = data.isnull().sum()
print(missing_values)
