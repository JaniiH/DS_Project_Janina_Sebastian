"""DS Projekt Janina_Sebastian."""

__authors__ = "Janina Höhn, Sebastian Kolb"
#------------------------------------------------------------------------------------------------------------------
# Imports necessary to run this py file.
# Pandas for data manipulation.
import pandas as pd
import operator

# Sklearn for splitting training and validation dataset.
from sklearn.model_selection import train_test_split

# Statsmodels for linear regression model.
import statsmodels.api as sm

# Suppress warnings.
import warnings
warnings.filterwarnings('ignore')

# Matplotlib and seaborn for plotting.
import matplotlib.pyplot as plt
plt.interactive = True
import seaborn as sns

# Create polynomial features and interaction terms automatically.
from sklearn.preprocessing import PolynomialFeatures
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Merge data set.
# Get base data bauenwohnen and bevoelkerung.
data_bauenwohnen = pd.read_excel(r'D:\Users\BKU\SebastianKolb\Desktop\DS\bauenwohnen.xls')
date_bevoelkerung = pd.read_excel(r'D:\Users\BKU\SebastianKolb\Desktop\DS\bevoelkerung.xls')

# Use code as primary key.
app_data = data_bauenwohnen.merge(right=date_bevoelkerung, how="inner", on="Codes")

# Select relevant attributes.
app_data = app_data[['Codes', 'Stadtteil_x', 'Bauen und Wohnen Wohnfläche in m² je Wohnung  2012',
                             'Bevölkerung Durchschnittsalter  2012', 'Bevölkerung Ausländerinnen und Ausländer in %  2012',
                             'Bevölkerung Deutsche mit Migrationshintergrund in %  2012',
                             'Bevölkerung Einpersonenhaushalte in %  2012', 'Bevölkerung Familien mit Kindern in %  2012']]

# Rename attributes.
app_data.columns = ['Codes', 'Stadtteil', 'Wohnfläche', 'Alter', 'Ausländer', 'Migrationshintergrund',
                    'Einpersonenhaushalt', 'Familie_Kinderhaushalt']

# Save merged data as excel.
export_excel = app_data.to_excel('Dataset_merged.xlsx')
#------------------------------------------------------------------------------------------------------------------
# Load data set.
# Adjust pd.read_excel according to where .xlsx is saved.
app_data = pd.read_excel(r'D:\Users\BKU\SebastianKolb\Desktop\DS\Dataset_merged.xlsx')
print('Data shape: ', app_data.shape)
app_data.head()

# Delete irrelevant attributes.
if 'Unnamed: 0' in app_data:
    del app_data['Unnamed: 0']
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Data understanding about app_data.
# Exploratory data analysis and data quality check.

# Distribution of target in histogram.
app_data['Wohnfläche'].astype(int).plot.hist()

# Overview on predictor variables.
app_data.info()

# Types of predictor variables.
print(app_data.dtypes.value_counts())

# Check number of unique categories in categorical predictor variables (text form).
print(app_data.select_dtypes('object').apply(pd.Series.nunique, axis=0))

# Identify outliers.
# Iterate through the attributes.
for col in app_data:
    print(app_data[col].describe())

# Function to calculate missing values by attribute.
def missing_values_table(df):
    # Total missing values.
    mis_val = df.isnull().sum()

    # Percentage of missing values.
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results.
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns.
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending.
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print summary of missing values.
    print("Your selected dataframe has " + str(df.shape[1]) +
           " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")

    # Return the data frame with missing information.
    return mis_val_table_ren_columns

# Missing values statistics.
missing_values = missing_values_table(app_data)
missing_values.head()
#------------------------------------------------------------------------------------------------------------------
# Examine correlations between predictor variables and Wohnfläche on app_data.
correlations_target = app_data.corr()['Wohnfläche'].sort_values()

# Display correlations.
print('Correlations to Wohnfläche:\n', correlations_target)

# Examine correlations between all variables.
correlations_target = app_data[list(correlations_target.index)]
correlations = correlations_target.corr()

# Heatmap of correlations.
plt.figure(figsize=(100, 20))
sns.heatmap(correlations, cmap=plt.cm.RdYlBu_r, vmin=-1, annot=False, vmax=1)
plt.title('Correlation Heatmap');

# Scatter plot of Wohnfläche and specific predictor variable.
app_data.plot.scatter('Alter', 'Wohnfläche', s=None, c=None)
app_data.plot.scatter('Ausländer', 'Wohnfläche', s=None, c=None)
app_data.plot.scatter('Migrationshintergrund', 'Wohnfläche', s=None, c=None)
app_data.plot.scatter('Einpersonenhaushalt', 'Wohnfläche', s=None, c=None)
app_data.plot.scatter('Familie_Kinderhaushalt', 'Wohnfläche', s=None, c=None)
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Modelling.
# Create training set and validation set.

# Define the target variable (dependent variable) as y.
y = app_data.Wohnfläche

# Split: 80% of observations in training set, 20% of observations in validation set.
# Fixed split (random_state is integer): When code executed multiple times, split remains the same.
x_train, x_test, y_train, y_test = train_test_split(app_data, y, test_size=0.2, random_state=1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Examine distribution of Wohnfläche.
# Training set.
y_train.value_counts()

# Validation set.
y_test.value_counts()

# Distribution of Wohnfläche in histogram: training (blue), validation set (orange).
y_train.astype(int).plot.hist()
y_test.astype(int).plot.hist()

# Delete Stadtteil and Codes for modelling.
del x_train['Stadtteil']
del x_test['Stadtteil']
del x_train['Codes']
del x_test['Codes']
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Modelling.
# Baseline linear regression model (statsmodel package) with single predictor.
# Chose predictor.
x_train_single = x_train['Ausländer']
x_test_single = x_test['Ausländer']

# Add constant so that intercept is included into regression function.
if 'const' not in x_train_single:
    x_train_single = sm.add_constant(x_train_single)
if 'const' not in x_test_single:
    x_test_single = sm.add_constant(x_test_single)

# Train baseline model.
baseline_model = sm.OLS(y_train, x_train_single)
result = baseline_model.fit()
# Show regression results.
print(result.summary())

# Predict y using the baseline model.
y_pred = result.predict(x_test_single)

plt.figure(figsize=(10, 10))
plt.scatter(x_train['Ausländer'].values.reshape(-1, 1), y_train, color='blue')  # Plot training set.
plt.scatter(x_test['Ausländer'].values.reshape(-1, 1), y_test, color='green')  # Plot validation set.
plt.plot(x_test_single['Ausländer'].values.reshape(-1, 1), y_pred, color='red')  # Plot prediction based on validation set.
plt.xlabel('Ausländer')
plt.ylabel('Wohnfläche')
plt.show()

# Compare prediction to actual output.
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Model 1: Linear regression model (statsmodel package) with multiple predictors.
x_train_multiple = x_train
x_test_multiple = x_test

# Drop the target from the training set.
if 'Wohnfläche' in x_train_multiple:
    x_train_multiple = x_train_multiple.drop(columns=['Wohnfläche'])

# Drop the target from the validation set.
if 'Wohnfläche' in x_test_multiple:
    x_test_multiple = x_test_multiple.drop(columns=['Wohnfläche'])

# Add constant so that intercept is included into regression function.
if 'const' not in x_train_multiple:
    x_train_multiple = sm.add_constant(x_train_multiple)

# Add constant to validation set to match number of columns.
if 'const' not in x_test_multiple:
    x_test_multiple = sm.add_constant(x_test_multiple)

# Train model 1.
model_one = sm.OLS(y_train, x_train_multiple)
result = model_one.fit()
# Show regression results of model 1.
print(result.summary())

# Plot partial regression plots.
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(result, fig=fig)
plt.show()

# Predict y using the model.
y_pred = result.predict(x_test_multiple)

# Compare prediction to actual output.
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Data preparation.
# Create polynomial features and interaction terms automatically by sklearn class PolynomialFeatures.
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html.

# Create new data sets for calculating polynomial features and interaction terms.
x_train_poly = x_train
variable_names = list(x_train_poly.columns)
x_test_poly = x_test

# Define Wohnfläche as target variable.
y_train_poly = y_train
y_test_poly = y_test
print(x_train_poly)

# Create the polynomial features with specified degree and interaction terms.
poly_transformer = PolynomialFeatures(degree=2)
# Train the polynomial features and interaction terms.
poly_transformer.fit(x_train_poly)
# Transform the new features.
x_train_poly = poly_transformer.transform(x_train_poly)
x_test_poly = poly_transformer.transform(x_test_poly)
print('Polynomial Features shape: ', x_train_poly.shape)

print(x_test_poly)
# Get names of polynomial features and interaction terms.
poly_transformer.get_feature_names(input_features=variable_names)

# Create a training set with polynomial features and interaction terms.
x_train_poly = pd.DataFrame(x_train_poly, columns=poly_transformer.get_feature_names(variable_names))
# Create a validation set with polynomial features and interaction terms.
x_test_poly = pd.DataFrame(x_test_poly, columns=poly_transformer.get_feature_names(variable_names))

# Define y_train_poly.
y_train_poly = x_train_poly['Wohnfläche']
# Define y_test_poly.
y_test_poly = x_test_poly['Wohnfläche']

print(x_train_poly)
print(x_test_poly)

# Rename  'Wohnfläche' to 'target' in training set.
x_train_poly = x_train_poly.rename(columns={'Wohnfläche': 'target'})
# Delete polynomial features and interaction terms with 'Wohnfläche' in training set.
x_train_poly = x_train_poly[x_train_poly.columns.drop(list(x_train_poly.filter(regex='Wohnfläche')))]
# Delete polynomial features and interaction terms with 'Wohnfläche' in validation set.
x_test_poly = x_test_poly[x_test_poly.columns.drop(list(x_test_poly.filter(regex='Wohnfläche')))]
# Rename 'target' to 'Wohnfläche' in training set.
x_train_poly = x_train_poly.rename(columns={'target': 'Wohnfläche'})

print(x_test_poly.shape)
print(x_train_poly.shape)
#------------------------------------------------------------------------------------------------------------------
# Evaluate effect of polynomial features and interaction terms.
# Find the correlations with Wohnfläche.
poly_corrs = x_train_poly.corr()['Wohnfläche'].sort_values()

# Display correlations with Wohnfläche.
print(poly_corrs)
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# Model 2: Linear regression model (statsmodel package) with polynomial and interaction terms as predictors.

# Drop the target from the training set.
if 'Wohnfläche' in x_train_poly:
    x_train_poly = x_train_poly.drop(columns=['Wohnfläche'])

# Add constant so that intercept is included into regression function.
if '1' not in x_train_poly:
    x_train_poly = sm.add_constant(x_train_poly)

# Add constant to validation set to match number of columns.
if '1' not in x_test_poly:
    x_test_poly = sm.add_constant(x_test_poly)

# Model 2a: Includes all polynomial and interaction terms as predictors.
# Train model 2a.
model_two_a = sm.OLS(y_train_poly, x_train_poly)
result = model_two_a.fit()
# Show regression results of model 2a.
print(result.summary())

# Plot partial regression plots.
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(result, fig=fig)
plt.show()

# Predict y using model 2a.
y_poly_pred = result.predict(x_test_poly)

# Compare prediction to actual output.
df = pd.DataFrame({'Actual': y_test_poly, 'Predicted': y_poly_pred})
df
#------------------------------------------------------------------------------------------------------------------
# Model 2b: Includes specific predictors.
x_train_poly_b = x_train_poly[['1', 'Ausländer', 'Ausländer^2']]
x_test_poly_b = x_test_poly[['1', 'Ausländer', 'Ausländer^2']]

# Train model 2b.
model_two_b = sm.OLS(y_train_poly, x_train_poly_b)
result = model_two_b.fit()
# Show regression results of model 2b.
print(result.summary())

# Plot partial regression plots.
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(result, fig=fig)
plt.show()

# Predict y using model 2b.
y_poly_pred = result.predict(x_test_poly_b)

# Compare prediction to actual output.
df = pd.DataFrame({'Actual': y_test_poly, 'Predicted': y_poly_pred})
df

# Plot regression function model 2b.
plt.scatter(x_train['Ausländer'], y_train, s=10)
plt.scatter(x_test['Ausländer'], y_test, s=10)
# Sort the values of x before line plot.
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_test_poly_b['Ausländer'], y_poly_pred), key=sort_axis)
x_test_poly_b['Ausländer'], y_poly_pred = zip(*sorted_zip)
plt.plot(x_test_poly_b['Ausländer'], y_poly_pred, color='red')
plt.xlabel('Ausländer')
plt.ylabel('Wohnfläche')
plt.show()


