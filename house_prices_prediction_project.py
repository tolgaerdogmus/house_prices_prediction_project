
#Business Problem¶
#
#Using a dataset of properties and house prices for each house, a machine learning project on the prices of different types of houses is intended to be realized.
#Dataset Story
#
#This dataset of residential homes in Ames, Iowa contains 79 explanatory variables. A contest on Kaggle. You can access the dataset and the competition page of the project from the kaggle website. The dataset belongs to a kaggle competition, therefore, there are two different csv files, train and test. In the test dataset, house prices are left blank and this. We try expect to estimate the values.
#
#38 Numeric Variables, 43 Categorical Variables, 1460 Observation
#
#    MSSubClass: Identifies the type of dwelling involved in the sale
#    MSZoning: Identifies the general zoning classification of the sale
#    LotFrontage: Linear feet of street connected to property
#    LotArea: Lot size in square feet
#    Street: Type of road access to property
#    Alley: Type of alley access to property
#    LotShape: General shape of property
#    LandContour: Flatness of the property
#    Utilities: Type of utilities available
#    LotConfig: Lot configuration
#    LandSlope: Slope of property
#    Neighborhood: Physical locations within Ames city limits
#    Condition1: Proximity to various conditions
#    Condition2: Proximity to various conditions (if more than one is present)
#    BldgType: Type of dwelling
#    HouseStyle: Style of dwelling
#    OverallQual: Rates the overall material and finish of the house
#    OverallCond: Rates the overall condition of the house
#    YearBuilt: Original construction date
#    YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#    RoofStyle: Type of roof
#    RoofMatl: Roof material
#    Exterior1st: Exterior covering on house
#    Exterior2nd: Exterior covering on house (if more than one material)
#    MasVnrType: Masonry veneer type
#    MasVnrArea: Masonry veneer area in square feet
#    ExterQual: Evaluates the quality of the material on the exterior
#    ExterCond: Evaluates the present condition of the material on the exterior
#    Foundation: Type of foundation
#    BsmtQual: Evaluates the height of the basement
#    BsmtCond: Evaluates the general condition of the basement
#    BsmtExposure: Refers to walkout or garden level walls
#    BsmtFinType1: Rating of basement finished area
#    BsmtFinSF1: Type 1 finished square feet
#    BsmtFinType2: Rating of basement finished area (if multiple types)
#    BsmtFinSF2: Type 2 finished square feet
#    BsmtUnfSF: Unfinished square feet of basement area
#    TotalBsmtSF: Total square feet of basement area
#    Heating: Type of heating
#    HeatingQC: Heating quality and condition
#    CentralAir: Central air conditioning
#    Electrical: Electrical system
#    1stFlrSF: First Floor square feet
#    2ndFlrSF: Second floor square feet
#    LowQualFinSF: Low quality finished square feet (all floors)
#    GrLivArea: Above grade (ground) living area square feet
#    BsmtFullBath: Basement full bathrooms
#    BsmtHalfBath: Basement half bathrooms
#    FullBath: Full bathrooms above grade
#    HalfBath: Half baths above grade
#    Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#    Kitchen: Kitchens above grade
#    KitchenQual: Kitchen quality
#    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#    Functional: Home functionality (Assume typical unless deductions are warranted)
#    Fireplaces: Number of fireplaces
#    FireplaceQu: Fireplace quality
#    GarageType: Garage location
#    GarageYrBlt: Year garage was built
#    GarageFinish: Interior finish of the garage
#    GarageCars: Size of garage in car capacity
#    GarageArea: Size of garage in square feet
#    GarageQual: Garage quality
#    GarageCond: Garage condition
#    PavedDrive: Paved driveway
#    WoodDeckSF: Wood deck area in square feet
#    OpenPorchSF: Open porch area in square feet
#    EnclosedPorch: Enclosed porch area in square feet
#    3SsnPorch: Three season porch area in square feet
#    ScreenPorch: Screen porch area in square feet
#    PoolArea: Pool area in square feet
#    PoolQC: Pool quality
#    Fence: Fence quality
#    MiscFeature: Miscellaneous feature not covered in other categories
#    MiscVal: $Value of miscellaneous feature
#    MoSold: Month Sold (MM)
#    YrSold: Year Sold (YYYY)
#    SaleType: Type of sale
#    SaleCondition: Condition of sale
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
import missingno as msno

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_test = pd.read_csv('Datasets/test.csv')
df_train = pd.read_csv('Datasets/train.csv')
df = pd.concat([df_test, df_train], axis=1).reset_index(drop=True)

df = pd.read_csv('Datasets/combined.csv', index_col=False)
df.drop("Unnamed: 0", axis=1, inplace=True)
df.dtypes
df.to_csv('Datasets/combined.csv', index=False)
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Duplicated Values #####################")
    print(dataframe.duplicated().sum())

    print("##################### Number of Unique Values #####################")
    print(df.nunique())

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    numeric_cols = dataframe.select_dtypes(include=['number'])  # sayısal değerlerin quantileıne bakar
    print(numeric_cols.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Check the type of the DataFrame and its columns
print(f"Type of df: {type(df)}")
print("DataFrame columns and types:")
print(df.dtypes)

# Function to identify categorical, numerical, and cardinal variables in the dataset.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    This function identifies the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Numeric-looking categorical variables are also included in categorical variables.

    Parameters:
    dataframe: The dataframe to analyze.
    cat_th: Threshold for the number of unique values below which variables are considered categorical (default is 10).
    car_th: Threshold for the number of unique values above which variables are considered cardinal (default is 20).

    Returns:
    cat_cols: List of categorical variable names.
    num_cols: List of numerical variable names.
    cat_but_car: List of cardinal (categorical but with many unique values) variable names.
    """
    print("Starting grab_col_names function")

    # Categorical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O" or pd.api.types.is_object_dtype(dataframe[col])]
    print("Identified categorical columns:", cat_cols)

    # Numerical but categorical
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and pd.api.types.is_numeric_dtype(dataframe[col])]
    print("Identified numerical but categorical columns:", num_but_cat)

    # Categorical but cardinal
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and (dataframe[col].dtype == "O" or pd.api.types.is_object_dtype(dataframe[col]))]
    print("Identified categorical but cardinal columns:", cat_but_car)

    # Combine categorical columns and remove cardinal columns
    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    print("Final categorical columns:", cat_cols)

    # Numerical columns
    num_cols = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print("Identified numerical columns:", num_cols)

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# Execute the function to identify and separate variable types.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)
print("Cardinal columns:", cat_but_car)

