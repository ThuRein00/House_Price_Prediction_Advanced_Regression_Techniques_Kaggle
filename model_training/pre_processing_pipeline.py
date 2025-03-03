from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,RobustScaler,MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# pipelines
num_pipeline = Pipeline([
('minmax_Scaler' , RobustScaler()),# pipeline for numeric columns
('num_imputer', SimpleImputer(fill_value=0.0)),


])

cat_pipeline_one_hot = Pipeline(steps = [('one_hot_encoder',
                                          OneHotEncoder(handle_unknown= 'ignore')), #categorical one hot pipeline
                                         ('imputer', SimpleImputer(fill_value = 'NA'))])

cat_pipeline_ordinal = Pipeline(steps = [('ordinal_encoder',OrdinalEncoder(categories ='auto',handle_unknown="use_encoded_value", unknown_value= -3)) #categorical ordinal pipeline
                                          
                                 ,('imputer', SimpleImputer(fill_value = 'NA'))])


# grouping columns for column transformer

num = ['LotFrontage','LotArea','House_age','Remod_age','BsmtFin','BsmtUnfSF','TotalBsmtSF','FlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','OpenSF','bedroom/totalRoomAboveGrade']+['MasVnrArea']


cat_ordinal = ['ExterQual' , 'ExterCond' , 'BsmtQual' , 'HeatingQC' ,'KitchenQual', 'GarageQual' ,'GarageCond',
               'BsmtExposure','BsmtFinType1' ,'BsmtFinType2','GarageFinish']+['FireplaceQu'] + ['MSSubClass','OverallQual','OverallCond','Bath','AbvGrd','TotRmsAbvGrd','Fireplaces','GarageCars']
cat_one_hot = ['MSZoning','LotShape','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
               'BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation','BsmtCond',
               'CentralAir','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition']+['Street', 'RoofMatl', 'Heating']


#full pipelines 
full_pipeline = ColumnTransformer(transformers = [
    ('num' , num_pipeline,num),
    ('cat_one_hot' , cat_pipeline_one_hot,cat_one_hot),
    ('cat_ordinal' , cat_pipeline_ordinal,cat_ordinal)],
    remainder = 'drop', n_jobs= -1)



