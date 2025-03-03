import numpy as np
def add_new_features(a): #for train set
    df = a
    df['LogSalePrice'] = np.log(df['SalePrice'])
    df['BsmtFin'] = df['BsmtFinSF1']+ df['BsmtFinSF2']
    df['House_age'] = df['YrSold'] - df['YearBuilt']
    df['Remod_age'] = df['YrSold'] - df['YearRemodAdd']
    df['FlrSF'] = df['1stFlrSF'] +df['2ndFlrSF'] 
    df['Bath'] = df['HalfBath']+df['FullBath']
    df['AbvGrd'] = df['KitchenAbvGr']+df['BedroomAbvGr']
    df['OpenSF'] = df['WoodDeckSF'] + df['OpenPorchSF']+ df['EnclosedPorch']+ df['3SsnPorch']+df['ScreenPorch']
    df['bedroom/totalRoomAboveGrade'] = df['BedroomAbvGr'] / df['TotRmsAbvGrd']
    df=df[~(df['LotFrontage']>300)]   
    # df=df[~(df['SalePrice']>700000)]
    # df=df[~(df['LotArea']>100000)]
    # df=df[~(df['MasVnrArea']>1250)]
    # df=df[~(df['BsmtFin']>4000)]
    # df=df[~(df['TotalBsmtSF']>3000)]
    # df=df[~(df['GarageArea']>1200)]
    # df=df[~(df['LogSalePrice']<10.75)]
    return df

def add_new_features_no_clean(a): #for test set
    df = a
    df['BsmtFin'] = df['BsmtFinSF1']+ df['BsmtFinSF2']
    df['House_age'] = df['YrSold'] - df['YearBuilt']
    df['Remod_age'] = df['YrSold'] - df['YearRemodAdd']
    df['FlrSF'] = df['1stFlrSF'] +df['2ndFlrSF'] 
    df['Bath'] = df['HalfBath']+df['FullBath']
    df['AbvGrd'] = df['KitchenAbvGr']+df['BedroomAbvGr']
    df['OpenSF'] = df['WoodDeckSF'] + df['OpenPorchSF']+ df['EnclosedPorch']+ df['3SsnPorch']+df['ScreenPorch']
    df['bedroom/totalRoomAboveGrade'] = df['BedroomAbvGr'] / df['TotRmsAbvGrd']
    return df