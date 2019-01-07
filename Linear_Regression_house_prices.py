import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd

train_data = pd.read_csv('./all/train.csv')

train_data.loc[train_data['MSZoning'] == 'RL', 'MSZoning'] = 1
train_data.loc[train_data['MSZoning'] == 'RM', 'MSZoning'] = 2
train_data.loc[train_data['MSZoning'] == 'C (all)', 'MSZoning'] = 3
train_data.loc[train_data['MSZoning'] == 'FV', 'MSZoning'] = 4
train_data.loc[train_data['MSZoning'] == 'RH', 'MSZoning'] = 5


train_data.loc[train_data['Street'] == 'Pave', 'Street'] = 1
train_data.loc[train_data['Street'] == 'Grvl', 'Street'] = 2

train_data.loc[train_data['Alley'] == 'Pave', 'Alley'] = 1
train_data.loc[train_data['Alley'] == 'Grvl', 'Alley'] = 2

train_data.loc[train_data['LotShape'] == 'Reg', 'LotShape'] = 1
train_data.loc[train_data['LotShape'] == 'IR1', 'LotShape'] = 2
train_data.loc[train_data['LotShape'] == 'IR2', 'LotShape'] = 3
train_data.loc[train_data['LotShape'] == 'IR3', 'LotShape'] = 4

train_data.loc[train_data['LandContour'] == 'Lvl', 'LandContour'] = 1
train_data.loc[train_data['LandContour'] == 'Bnk', 'LandContour'] = 2
train_data.loc[train_data['LandContour'] == 'Low', 'LandContour'] = 3
train_data.loc[train_data['LandContour'] == 'HLS', 'LandContour'] = 4

train_data.loc[train_data['Utilities'] == 'AllPub', 'Utilities'] = 1
train_data.loc[train_data['Utilities'] == 'NoSeWa', 'Utilities'] = 2

train_data.loc[train_data['LotConfig'] == 'Inside', 'LotConfig'] = 1
train_data.loc[train_data['LotConfig'] == 'FR2', 'LotConfig'] = 2
train_data.loc[train_data['LotConfig'] == 'Corner', 'LotConfig'] = 3
train_data.loc[train_data['LotConfig'] == 'CulDSac', 'LotConfig'] = 4
train_data.loc[train_data['LotConfig'] == 'FR3', 'LotConfig'] = 5

train_data.loc[train_data['LandSlope'] == 'Gtl', 'LandSlope'] = 1
train_data.loc[train_data['LandSlope'] == 'Mod', 'LandSlope'] = 2
train_data.loc[train_data['LandSlope'] == 'Sev', 'LandSlope'] = 3

train_data.loc[train_data['Neighborhood'] == 'CollgCr', 'Neighborhood'] = 1
train_data.loc[train_data['Neighborhood'] == 'Veenker', 'Neighborhood'] = 2
train_data.loc[train_data['Neighborhood'] == 'Crawfor', 'Neighborhood'] = 3
train_data.loc[train_data['Neighborhood'] == 'NoRidge', 'Neighborhood'] = 4
train_data.loc[train_data['Neighborhood'] == 'NoRidge', 'Neighborhood'] = 5
train_data.loc[train_data['Neighborhood'] == 'Mitchel', 'Neighborhood'] = 6
train_data.loc[train_data['Neighborhood'] == 'Somerst', 'Neighborhood'] = 7
train_data.loc[train_data['Neighborhood'] == 'NWAmes', 'Neighborhood'] = 8
train_data.loc[train_data['Neighborhood'] == 'OldTown', 'Neighborhood'] = 9
train_data.loc[train_data['Neighborhood'] == 'BrkSide', 'Neighborhood'] = 10
train_data.loc[train_data['Neighborhood'] == 'Sawyer', 'Neighborhood'] = 11
train_data.loc[train_data['Neighborhood'] == 'NridgHt', 'Neighborhood'] = 12
train_data.loc[train_data['Neighborhood'] == 'NAmes', 'Neighborhood'] = 13
train_data.loc[train_data['Neighborhood'] == 'SawyerW', 'Neighborhood'] = 14
train_data.loc[train_data['Neighborhood'] == 'IDOTRR', 'Neighborhood'] = 15
train_data.loc[train_data['Neighborhood'] == 'MeadowV', 'Neighborhood'] = 16
train_data.loc[train_data['Neighborhood'] == 'Edwards', 'Neighborhood'] = 17
train_data.loc[train_data['Neighborhood'] == 'Timber', 'Neighborhood'] = 18
train_data.loc[train_data['Neighborhood'] == 'Gilbert', 'Neighborhood'] = 19
train_data.loc[train_data['Neighborhood'] == 'StoneBr', 'Neighborhood'] = 20
train_data.loc[train_data['Neighborhood'] == 'ClearCr', 'Neighborhood'] = 21
train_data.loc[train_data['Neighborhood'] == 'NPkVill', 'Neighborhood'] = 22
train_data.loc[train_data['Neighborhood'] == 'Blmngtn', 'Neighborhood'] = 23
train_data.loc[train_data['Neighborhood'] == 'BrDale', 'Neighborhood'] = 24
train_data.loc[train_data['Neighborhood'] == 'SWISU', 'Neighborhood'] = 25
train_data.loc[train_data['Neighborhood'] == 'Blueste', 'Neighborhood'] = 26

train_data.loc[train_data['Condition1'] == 'Norm', 'Condition1'] = 1
train_data.loc[train_data['Condition1'] == 'Feedr', 'Condition1'] = 2
train_data.loc[train_data['Condition1'] == 'PosN', 'Condition1'] = 3
train_data.loc[train_data['Condition1'] == 'Artery', 'Condition1'] = 4
train_data.loc[train_data['Condition1'] == 'RRAe', 'Condition1'] = 5
train_data.loc[train_data['Condition1'] == 'RRNn', 'Condition1'] = 6
train_data.loc[train_data['Condition1'] == 'RRAn', 'Condition1'] = 7
train_data.loc[train_data['Condition1'] == 'PosA', 'Condition1'] = 8
train_data.loc[train_data['Condition1'] == 'RRNe', 'Condition1'] = 9

train_data.loc[train_data['Condition2'] == 'Norm', 'Condition2'] = 1
train_data.loc[train_data['Condition2'] == 'Feedr', 'Condition2'] = 2
train_data.loc[train_data['Condition2'] == 'PosN', 'Condition2'] = 3
train_data.loc[train_data['Condition2'] == 'Artery', 'Condition2'] = 4
train_data.loc[train_data['Condition2'] == 'RRAe', 'Condition2'] = 5
train_data.loc[train_data['Condition2'] == 'RRNn', 'Condition2'] = 6
train_data.loc[train_data['Condition2'] == 'RRAn', 'Condition2'] = 7
train_data.loc[train_data['Condition2'] == 'PosA', 'Condition2'] = 8

train_data.loc[train_data['BldgType'] == '1Fam', 'BldgType'] = 1
train_data.loc[train_data['BldgType'] == '2fmCon', 'BldgType'] = 2
train_data.loc[train_data['BldgType'] == 'Duplex', 'BldgType'] = 3
train_data.loc[train_data['BldgType'] == 'TwnhsE', 'BldgType'] = 4
train_data.loc[train_data['BldgType'] == 'Twnhs', 'BldgType'] = 5

train_data.loc[train_data['HouseStyle'] == '2Story', 'HouseStyle'] = 1
train_data.loc[train_data['HouseStyle'] == '1Story', 'HouseStyle'] = 2
train_data.loc[train_data['HouseStyle'] == '1.5Fin', 'HouseStyle'] = 3
train_data.loc[train_data['HouseStyle'] == '1.5Unf', 'HouseStyle'] = 4
train_data.loc[train_data['HouseStyle'] == 'SFoyer', 'HouseStyle'] = 5
train_data.loc[train_data['HouseStyle'] == 'SLvl', 'HouseStyle'] = 6
train_data.loc[train_data['HouseStyle'] == '2.5Unf', 'HouseStyle'] = 7
train_data.loc[train_data['HouseStyle'] == '2.5Fin', 'HouseStyle'] = 8

train_data.loc[train_data['RoofStyle'] == 'Gable', 'RoofStyle'] = 1
train_data.loc[train_data['RoofStyle'] == 'Hip', 'RoofStyle'] = 2
train_data.loc[train_data['RoofStyle'] == 'Gambrel', 'RoofStyle'] = 3
train_data.loc[train_data['RoofStyle'] == 'Mansard', 'RoofStyle'] = 4
train_data.loc[train_data['RoofStyle'] == 'Flat', 'RoofStyle'] = 5
train_data.loc[train_data['RoofStyle'] == 'Shed', 'RoofStyle'] = 6

train_data.loc[train_data['RoofMatl'] == 'CompShg', 'RoofMatl'] = 1
train_data.loc[train_data['RoofMatl'] == 'WdShngl', 'RoofMatl'] = 2
train_data.loc[train_data['RoofMatl'] == 'Metal', 'RoofMatl'] = 3
train_data.loc[train_data['RoofMatl'] == 'WdShake', 'RoofMatl'] = 4
train_data.loc[train_data['RoofMatl'] == 'Membran', 'RoofMatl'] = 5
train_data.loc[train_data['RoofMatl'] == 'Tar&Grv', 'RoofMatl'] = 6
train_data.loc[train_data['RoofMatl'] == 'Roll', 'RoofMatl'] = 7
train_data.loc[train_data['RoofMatl'] == 'ClyTile', 'RoofMatl'] = 8

train_data.loc[train_data['Exterior1st'] == 'VinylSd', 'Exterior1st'] = 1
train_data.loc[train_data['Exterior1st'] == 'MetalSd', 'Exterior1st'] = 2
train_data.loc[train_data['Exterior1st'] == 'Wd Sdng', 'Exterior1st'] = 3
train_data.loc[train_data['Exterior1st'] == 'HdBoard', 'Exterior1st'] = 4
train_data.loc[train_data['Exterior1st'] == 'BrkFace', 'Exterior1st'] = 5
train_data.loc[train_data['Exterior1st'] == 'WdShing', 'Exterior1st'] = 6
train_data.loc[train_data['Exterior1st'] == 'CemntBd', 'Exterior1st'] = 7
train_data.loc[train_data['Exterior1st'] == 'Plywood', 'Exterior1st'] = 8
train_data.loc[train_data['Exterior1st'] == 'AsbShng', 'Exterior1st'] = 9
train_data.loc[train_data['Exterior1st'] == 'Stucco', 'Exterior1st'] = 10
train_data.loc[train_data['Exterior1st'] == 'BrkComm', 'Exterior1st'] = 11
train_data.loc[train_data['Exterior1st'] == 'AsphShn', 'Exterior1st'] = 12
train_data.loc[train_data['Exterior1st'] == 'Stone', 'Exterior1st'] = 13
train_data.loc[train_data['Exterior1st'] == 'ImStucc', 'Exterior1st'] = 14
train_data.loc[train_data['Exterior1st'] == 'CBlock', 'Exterior1st'] = 15

train_data.loc[train_data['Exterior2nd'] == 'VinylSd', 'Exterior2nd'] = 1
train_data.loc[train_data['Exterior2nd'] == 'MetalSd', 'Exterior2nd'] = 2
train_data.loc[train_data['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 3
train_data.loc[train_data['Exterior2nd'] == 'HdBoard', 'Exterior2nd'] = 4
train_data.loc[train_data['Exterior2nd'] == 'Plywood', 'Exterior2nd'] = 5
train_data.loc[train_data['Exterior2nd'] == 'Wd Sdng', 'Exterior2nd'] = 6
train_data.loc[train_data['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 7
train_data.loc[train_data['Exterior2nd'] == 'BrkFace', 'Exterior2nd'] = 8
train_data.loc[train_data['Exterior2nd'] == 'Stucco', 'Exterior2nd'] = 9
train_data.loc[train_data['Exterior2nd'] == 'AsbShng', 'Exterior2nd'] = 10
train_data.loc[train_data['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 11
train_data.loc[train_data['Exterior2nd'] == 'ImStucc', 'Exterior2nd'] = 12
train_data.loc[train_data['Exterior2nd'] == 'AsphShn', 'Exterior2nd'] = 13
train_data.loc[train_data['Exterior2nd'] == 'Stone', 'Exterior2nd'] = 14
train_data.loc[train_data['Exterior2nd'] == 'Other', 'Exterior2nd'] = 15
train_data.loc[train_data['Exterior2nd'] == 'CBlock', 'Exterior2nd'] = 16

train_data.loc[train_data['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 1
train_data.loc[train_data['MasVnrType'] == 'None', 'MasVnrType'] = 2
train_data.loc[train_data['MasVnrType'] == 'Stone', 'MasVnrType'] = 3
train_data.loc[train_data['MasVnrType'] == 'BrkCmn', 'MasVnrType'] = 4

train_data.loc[train_data['ExterQual'] == 'Gd', 'ExterQual'] = 1
train_data.loc[train_data['ExterQual'] == 'TA', 'ExterQual'] = 2
train_data.loc[train_data['ExterQual'] == 'Ex', 'ExterQual'] = 3
train_data.loc[train_data['ExterQual'] == 'Fa', 'ExterQual'] = 4

train_data.loc[train_data['ExterCond'] == 'TA', 'ExterCond'] = 1
train_data.loc[train_data['ExterCond'] == 'Gd', 'ExterCond'] = 2
train_data.loc[train_data['ExterCond'] == 'Fa', 'ExterCond'] = 3
train_data.loc[train_data['ExterCond'] == 'Po', 'ExterCond'] = 4
train_data.loc[train_data['ExterCond'] == 'Ex', 'ExterCond'] = 5

train_data.loc[train_data['Foundation'] == 'PConc', 'Foundation'] = 1
train_data.loc[train_data['Foundation'] == 'CBlock', 'Foundation'] = 2
train_data.loc[train_data['Foundation'] == 'BrkTil', 'Foundation'] = 3
train_data.loc[train_data['Foundation'] == 'Wood', 'Foundation'] = 4
train_data.loc[train_data['Foundation'] == 'Slab', 'Foundation'] = 5
train_data.loc[train_data['Foundation'] == 'Stone', 'Foundation'] = 6

train_data.loc[train_data['BsmtQual'] == 'Gd', 'BsmtQual'] = 1
train_data.loc[train_data['BsmtQual'] == 'TA', 'BsmtQual'] = 2
train_data.loc[train_data['BsmtQual'] == 'Ex', 'BsmtQual'] = 3
train_data.loc[train_data['BsmtQual'] == 'Fa', 'BsmtQual'] = 4

train_data.loc[train_data['BsmtCond'] == 'TA', 'BsmtCond'] = 1
train_data.loc[train_data['BsmtCond'] == 'Gd', 'BsmtCond'] = 2
train_data.loc[train_data['BsmtCond'] == 'Fa', 'BsmtCond'] = 3
train_data.loc[train_data['BsmtCond'] == 'Po', 'BsmtCond'] = 4

train_data.loc[train_data['BsmtExposure'] == 'No', 'BsmtExposure'] = 1
train_data.loc[train_data['BsmtExposure'] == 'Gd', 'BsmtExposure'] = 2
train_data.loc[train_data['BsmtExposure'] == 'Mn', 'BsmtExposure'] = 3
train_data.loc[train_data['BsmtExposure'] == 'Av', 'BsmtExposure'] = 4

train_data.loc[train_data['BsmtFinType1'] == 'GLQ', 'BsmtFinType1'] = 1
train_data.loc[train_data['BsmtFinType1'] == 'ALQ', 'BsmtFinType1'] = 2
train_data.loc[train_data['BsmtFinType1'] == 'Unf', 'BsmtFinType1'] = 3
train_data.loc[train_data['BsmtFinType1'] == 'Rec', 'BsmtFinType1'] = 4
train_data.loc[train_data['BsmtFinType1'] == 'BLQ', 'BsmtFinType1'] = 5
train_data.loc[train_data['BsmtFinType1'] == 'LwQ', 'BsmtFinType1'] = 6

train_data.loc[train_data['BsmtFinType2'] == 'GLQ', 'BsmtFinType2'] = 1
train_data.loc[train_data['BsmtFinType2'] == 'ALQ', 'BsmtFinType2'] = 2
train_data.loc[train_data['BsmtFinType2'] == 'Unf', 'BsmtFinType2'] = 3
train_data.loc[train_data['BsmtFinType2'] == 'Rec', 'BsmtFinType2'] = 4
train_data.loc[train_data['BsmtFinType2'] == 'BLQ', 'BsmtFinType2'] = 5
train_data.loc[train_data['BsmtFinType2'] == 'LwQ', 'BsmtFinType2'] = 6

train_data.loc[train_data['Heating'] == 'GasA', 'Heating'] = 1
train_data.loc[train_data['Heating'] == 'GasW', 'Heating'] = 2
train_data.loc[train_data['Heating'] == 'Wall', 'Heating'] = 3
train_data.loc[train_data['Heating'] == 'OthW', 'Heating'] = 4
train_data.loc[train_data['Heating'] == 'Floor', 'Heating'] = 5
train_data.loc[train_data['Heating'] == 'Grav', 'Heating'] = 6

train_data.loc[train_data['HeatingQC'] == 'Ex', 'HeatingQC'] = 1
train_data.loc[train_data['HeatingQC'] == 'Gd', 'HeatingQC'] = 2
train_data.loc[train_data['HeatingQC'] == 'TA', 'HeatingQC'] = 3
train_data.loc[train_data['HeatingQC'] == 'Fa', 'HeatingQC'] = 4
train_data.loc[train_data['HeatingQC'] == 'Po', 'HeatingQC'] = 5

train_data.loc[train_data['CentralAir'] == 'Y', 'CentralAir'] = 1
train_data.loc[train_data['CentralAir'] == 'N', 'CentralAir'] = 0

train_data.loc[train_data['Electrical'] == 'SBrkr', 'Electrical'] = 1
train_data.loc[train_data['Electrical'] == 'FuseF', 'Electrical'] = 2
train_data.loc[train_data['Electrical'] == 'FuseA', 'Electrical'] = 3
train_data.loc[train_data['Electrical'] == 'FuseP', 'Electrical'] = 4
train_data.loc[train_data['Electrical'] == 'Mix', 'Electrical'] = 5

train_data.loc[train_data['KitchenQual'] == 'Gd', 'KitchenQual'] = 1
train_data.loc[train_data['KitchenQual'] == 'TA', 'KitchenQual'] = 2
train_data.loc[train_data['KitchenQual'] == 'Ex', 'KitchenQual'] = 3
train_data.loc[train_data['KitchenQual'] == 'Fa', 'KitchenQual'] = 4

train_data.loc[train_data['Functional'] == 'Typ', 'Functional'] = 1
train_data.loc[train_data['Functional'] == 'Min1', 'Functional'] = 2
train_data.loc[train_data['Functional'] == 'Maj1', 'Functional'] = 3
train_data.loc[train_data['Functional'] == 'Min2', 'Functional'] = 4
train_data.loc[train_data['Functional'] == 'Mod', 'Functional'] = 5
train_data.loc[train_data['Functional'] == 'Maj2', 'Functional'] = 6
train_data.loc[train_data['Functional'] == 'Sev', 'Functional'] = 7

train_data.loc[train_data['FireplaceQu'] == 'TA', 'FireplaceQu'] = 1
train_data.loc[train_data['FireplaceQu'] == 'Gd', 'FireplaceQu'] = 2
train_data.loc[train_data['FireplaceQu'] == 'Fa', 'FireplaceQu'] = 3
train_data.loc[train_data['FireplaceQu'] == 'Ex', 'FireplaceQu'] = 4
train_data.loc[train_data['FireplaceQu'] == 'Po', 'FireplaceQu'] = 5

train_data.loc[train_data['GarageType'] == 'Attchd', 'GarageType'] = 1
train_data.loc[train_data['GarageType'] == 'Detchd', 'GarageType'] = 2
train_data.loc[train_data['GarageType'] == 'BuiltIn', 'GarageType'] = 3
train_data.loc[train_data['GarageType'] == 'CarPort', 'GarageType'] = 4
train_data.loc[train_data['GarageType'] == 'Basment', 'GarageType'] = 5
train_data.loc[train_data['GarageType'] == '2Types', 'GarageType'] = 6

train_data.loc[train_data['GarageFinish'] == 'RFn', 'GarageFinish'] = 1
train_data.loc[train_data['GarageFinish'] == 'Unf', 'GarageFinish'] = 2
train_data.loc[train_data['GarageFinish'] == 'Fin', 'GarageFinish'] = 3

train_data.loc[train_data['GarageQual'] == 'TA', 'GarageQual'] = 1
train_data.loc[train_data['GarageQual'] == 'Fa', 'GarageQual'] = 2
train_data.loc[train_data['GarageQual'] == 'Gd', 'GarageQual'] = 3
train_data.loc[train_data['GarageQual'] == 'Ex', 'GarageQual'] = 4
train_data.loc[train_data['GarageQual'] == 'Po', 'GarageQual'] = 5

train_data.loc[train_data['GarageCond'] == 'TA', 'GarageCond'] = 1
train_data.loc[train_data['GarageCond'] == 'Fa', 'GarageCond'] = 2
train_data.loc[train_data['GarageCond'] == 'Gd', 'GarageCond'] = 3
train_data.loc[train_data['GarageCond'] == 'Ex', 'GarageCond'] = 4
train_data.loc[train_data['GarageCond'] == 'Po', 'GarageCond'] = 5

train_data.loc[train_data['PavedDrive'] == 'Y', 'PavedDrive'] = 1
train_data.loc[train_data['PavedDrive'] == 'P', 'PavedDrive'] = 2
train_data.loc[train_data['PavedDrive'] == 'N', 'PavedDrive'] = 0

train_data.loc[train_data['PoolQC'] == 'Ex', 'PoolQC'] = 1
train_data.loc[train_data['PoolQC'] == 'Fa', 'PoolQC'] = 2
train_data.loc[train_data['PoolQC'] == 'Gd', 'PoolQC'] = 3

train_data.loc[train_data['Fence'] == 'MnPrv', 'Fence'] = 1
train_data.loc[train_data['Fence'] == 'GdWo', 'Fence'] = 2
train_data.loc[train_data['Fence'] == 'GdPrv', 'Fence'] = 3
train_data.loc[train_data['Fence'] == 'MnWw', 'Fence'] = 4

train_data.loc[train_data['MiscFeature'] == 'Shed', 'MiscFeature'] = 1
train_data.loc[train_data['MiscFeature'] == 'Gar2', 'MiscFeature'] = 2
train_data.loc[train_data['MiscFeature'] == 'Othr', 'MiscFeature'] = 3
train_data.loc[train_data['MiscFeature'] == 'TenC', 'MiscFeature'] = 4

train_data.loc[train_data['SaleType'] == 'WD', 'SaleType'] = 1
train_data.loc[train_data['SaleType'] == 'New', 'SaleType'] = 2
train_data.loc[train_data['SaleType'] == 'COD', 'SaleType'] = 3
train_data.loc[train_data['SaleType'] == 'ConLD', 'SaleType'] = 4
train_data.loc[train_data['SaleType'] == 'ConLI', 'SaleType'] = 5
train_data.loc[train_data['SaleType'] == 'CWD', 'SaleType'] = 6
train_data.loc[train_data['SaleType'] == 'ConLw', 'SaleType'] = 7
train_data.loc[train_data['SaleType'] == 'Con', 'SaleType'] = 8
train_data.loc[train_data['SaleType'] == 'Oth', 'SaleType'] = 9

train_data.loc[train_data['SaleCondition'] == 'Normal', 'SaleCondition'] = 1
train_data.loc[train_data['SaleCondition'] == 'Abnorml', 'SaleCondition'] = 2
train_data.loc[train_data['SaleCondition'] == 'Partial', 'SaleCondition'] = 3
train_data.loc[train_data['SaleCondition'] == 'AdjLand', 'SaleCondition'] = 4
train_data.loc[train_data['SaleCondition'] == 'Alloca', 'SaleCondition'] = 5
train_data.loc[train_data['SaleCondition'] == 'Family', 'SaleCondition'] = 6

train_data_processed = train_data.fillna(0)

train_labels = train_data_processed['SalePrice']

train_data_processed = train_data_processed[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']]

train_data_processed = train_data_processed.values
train_labels = train_labels.values

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(79, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu'))
model.add(layers.Dense(79, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu'))
model.add(layers.Dense(79, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu'))
model.add(layers.Dense(79, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu'))
model.add(layers.Dense(79, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu'))
model.add(layers.Dense(79, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu'))
#model.add(layers.Dense(79, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer=tf.train.AdadeltaOptimizer(0.2),
              loss=['mean_squared_logarithmic_error'],       # mean squared error
              metrics=['mae'])

model.fit(train_data_processed, train_labels, epochs=1200, batch_size=32)

test_data = pd.read_csv('all/test.csv')

test_data.loc[test_data['MSZoning'] == 'nan', 'MSZoning'] = 0
test_data.loc[test_data['MSZoning'] == 'RL', 'MSZoning'] = 1
test_data.loc[test_data['MSZoning'] == 'RM', 'MSZoning'] = 2
test_data.loc[test_data['MSZoning'] == 'C (all)', 'MSZoning'] = 3
test_data.loc[test_data['MSZoning'] == 'FV', 'MSZoning'] = 4
test_data.loc[test_data['MSZoning'] == 'RH', 'MSZoning'] = 5

#test_data.loc[test_data['Street'] == 'nan', 'Street'] = 0
test_data.loc[test_data['Street'] == 'Pave', 'Street'] = 1
test_data.loc[test_data['Street'] == 'Grvl', 'Street'] = 2

test_data.loc[test_data['Alley'] == 'nan', 'Alley'] = 0
test_data.loc[test_data['Alley'] == 'Pave', 'Alley'] = 1
test_data.loc[test_data['Alley'] == 'Grvl', 'Alley'] = 2

test_data.loc[test_data['LotShape'] == 'nan', 'LotShape'] = 0
test_data.loc[test_data['LotShape'] == 'Reg', 'LotShape'] = 1
test_data.loc[test_data['LotShape'] == 'IR1', 'LotShape'] = 2
test_data.loc[test_data['LotShape'] == 'IR2', 'LotShape'] = 3
test_data.loc[test_data['LotShape'] == 'IR3', 'LotShape'] = 4

test_data.loc[test_data['LandContour'] == 'nan', 'LandContour'] = 0
test_data.loc[test_data['LandContour'] == 'Lvl', 'LandContour'] = 1
test_data.loc[test_data['LandContour'] == 'Bnk', 'LandContour'] = 2
test_data.loc[test_data['LandContour'] == 'Low', 'LandContour'] = 3
test_data.loc[test_data['LandContour'] == 'HLS', 'LandContour'] = 4

test_data.loc[test_data['Utilities'] == 'nan', 'Utilities'] = 0
test_data.loc[test_data['Utilities'] == 'AllPub', 'Utilities'] = 1
test_data.loc[test_data['Utilities'] == 'NoSeWa', 'Utilities'] = 2

test_data.loc[test_data['LotConfig'] == 'nan', 'LotConfig'] = 0
test_data.loc[test_data['LotConfig'] == 'Inside', 'LotConfig'] = 1
test_data.loc[test_data['LotConfig'] == 'FR2', 'LotConfig'] = 2
test_data.loc[test_data['LotConfig'] == 'Corner', 'LotConfig'] = 3
test_data.loc[test_data['LotConfig'] == 'CulDSac', 'LotConfig'] = 4
test_data.loc[test_data['LotConfig'] == 'FR3', 'LotConfig'] = 5

test_data.loc[test_data['LandSlope'] == 'nan', 'LandSlope'] = 0
test_data.loc[test_data['LandSlope'] == 'Gtl', 'LandSlope'] = 1
test_data.loc[test_data['LandSlope'] == 'Mod', 'LandSlope'] = 2
test_data.loc[test_data['LandSlope'] == 'Sev', 'LandSlope'] = 3

test_data.loc[test_data['Neighborhood'] == 'nan', 'Neighborhood'] = 0
test_data.loc[test_data['Neighborhood'] == 'CollgCr', 'Neighborhood'] = 1
test_data.loc[test_data['Neighborhood'] == 'Veenker', 'Neighborhood'] = 2
test_data.loc[test_data['Neighborhood'] == 'Crawfor', 'Neighborhood'] = 3
test_data.loc[test_data['Neighborhood'] == 'NoRidge', 'Neighborhood'] = 4
test_data.loc[test_data['Neighborhood'] == 'NoRidge', 'Neighborhood'] = 5
test_data.loc[test_data['Neighborhood'] == 'Mitchel', 'Neighborhood'] = 6
test_data.loc[test_data['Neighborhood'] == 'Somerst', 'Neighborhood'] = 7
test_data.loc[test_data['Neighborhood'] == 'NWAmes', 'Neighborhood'] = 8
test_data.loc[test_data['Neighborhood'] == 'OldTown', 'Neighborhood'] = 9
test_data.loc[test_data['Neighborhood'] == 'BrkSide', 'Neighborhood'] = 10
test_data.loc[test_data['Neighborhood'] == 'Sawyer', 'Neighborhood'] = 11
test_data.loc[test_data['Neighborhood'] == 'NridgHt', 'Neighborhood'] = 12
test_data.loc[test_data['Neighborhood'] == 'NAmes', 'Neighborhood'] = 13
test_data.loc[test_data['Neighborhood'] == 'SawyerW', 'Neighborhood'] = 14
test_data.loc[test_data['Neighborhood'] == 'IDOTRR', 'Neighborhood'] = 15
test_data.loc[test_data['Neighborhood'] == 'MeadowV', 'Neighborhood'] = 16
test_data.loc[test_data['Neighborhood'] == 'Edwards', 'Neighborhood'] = 17
test_data.loc[test_data['Neighborhood'] == 'Timber', 'Neighborhood'] = 18
test_data.loc[test_data['Neighborhood'] == 'Gilbert', 'Neighborhood'] = 19
test_data.loc[test_data['Neighborhood'] == 'StoneBr', 'Neighborhood'] = 20
test_data.loc[test_data['Neighborhood'] == 'ClearCr', 'Neighborhood'] = 21
test_data.loc[test_data['Neighborhood'] == 'NPkVill', 'Neighborhood'] = 22
test_data.loc[test_data['Neighborhood'] == 'Blmngtn', 'Neighborhood'] = 23
test_data.loc[test_data['Neighborhood'] == 'BrDale', 'Neighborhood'] = 24
test_data.loc[test_data['Neighborhood'] == 'SWISU', 'Neighborhood'] = 25
test_data.loc[test_data['Neighborhood'] == 'Blueste', 'Neighborhood'] = 26

test_data.loc[test_data['Condition1'] == 'nan', 'Condition1'] = 0
test_data.loc[test_data['Condition1'] == 'Norm', 'Condition1'] = 1
test_data.loc[test_data['Condition1'] == 'Feedr', 'Condition1'] = 2
test_data.loc[test_data['Condition1'] == 'PosN', 'Condition1'] = 3
test_data.loc[test_data['Condition1'] == 'Artery', 'Condition1'] = 4
test_data.loc[test_data['Condition1'] == 'RRAe', 'Condition1'] = 5
test_data.loc[test_data['Condition1'] == 'RRNn', 'Condition1'] = 6
test_data.loc[test_data['Condition1'] == 'RRAn', 'Condition1'] = 7
test_data.loc[test_data['Condition1'] == 'PosA', 'Condition1'] = 8
test_data.loc[test_data['Condition1'] == 'RRNe', 'Condition1'] = 9

test_data.loc[test_data['Condition2'] == 'nan', 'Condition2'] = 0
test_data.loc[test_data['Condition2'] == 'Norm', 'Condition2'] = 1
test_data.loc[test_data['Condition2'] == 'Feedr', 'Condition2'] = 2
test_data.loc[test_data['Condition2'] == 'PosN', 'Condition2'] = 3
test_data.loc[test_data['Condition2'] == 'Artery', 'Condition2'] = 4
test_data.loc[test_data['Condition2'] == 'RRAe', 'Condition2'] = 5
test_data.loc[test_data['Condition2'] == 'RRNn', 'Condition2'] = 6
test_data.loc[test_data['Condition2'] == 'RRAn', 'Condition2'] = 7
test_data.loc[test_data['Condition2'] == 'PosA', 'Condition2'] = 8

test_data.loc[test_data['BldgType'] == 'nan', 'BldgType'] = 0
test_data.loc[test_data['BldgType'] == '1Fam', 'BldgType'] = 1
test_data.loc[test_data['BldgType'] == '2fmCon', 'BldgType'] = 2
test_data.loc[test_data['BldgType'] == 'Duplex', 'BldgType'] = 3
test_data.loc[test_data['BldgType'] == 'TwnhsE', 'BldgType'] = 4
test_data.loc[test_data['BldgType'] == 'Twnhs', 'BldgType'] = 5

test_data.loc[test_data['HouseStyle'] == 'nan', 'HouseStyle'] = 0
test_data.loc[test_data['HouseStyle'] == '2Story', 'HouseStyle'] = 1
test_data.loc[test_data['HouseStyle'] == '1Story', 'HouseStyle'] = 2
test_data.loc[test_data['HouseStyle'] == '1.5Fin', 'HouseStyle'] = 3
test_data.loc[test_data['HouseStyle'] == '1.5Unf', 'HouseStyle'] = 4
test_data.loc[test_data['HouseStyle'] == 'SFoyer', 'HouseStyle'] = 5
test_data.loc[test_data['HouseStyle'] == 'SLvl', 'HouseStyle'] = 6
test_data.loc[test_data['HouseStyle'] == '2.5Unf', 'HouseStyle'] = 7
#test_data.loc[test_data['HouseStyle'] == '2.5Fin', 'HouseStyle'] = 8

#test_data.loc[test_data['RoofStyle'] == 'nan', 'RoofStyle'] = 0
test_data.loc[test_data['RoofStyle'] == 'Gable', 'RoofStyle'] = 1
test_data.loc[test_data['RoofStyle'] == 'Hip', 'RoofStyle'] = 2
test_data.loc[test_data['RoofStyle'] == 'Gambrel', 'RoofStyle'] = 3
test_data.loc[test_data['RoofStyle'] == 'Mansard', 'RoofStyle'] = 4
test_data.loc[test_data['RoofStyle'] == 'Flat', 'RoofStyle'] = 5
test_data.loc[test_data['RoofStyle'] == 'Shed', 'RoofStyle'] = 6

test_data.loc[test_data['RoofMatl'] == 'nan', 'RoofMatl'] = 0
test_data.loc[test_data['RoofMatl'] == 'CompShg', 'RoofMatl'] = 1
test_data.loc[test_data['RoofMatl'] == 'WdShngl', 'RoofMatl'] = 2
test_data.loc[test_data['RoofMatl'] == 'Metal', 'RoofMatl'] = 3
test_data.loc[test_data['RoofMatl'] == 'WdShake', 'RoofMatl'] = 4
test_data.loc[test_data['RoofMatl'] == 'Membran', 'RoofMatl'] = 5
test_data.loc[test_data['RoofMatl'] == 'Tar&Grv', 'RoofMatl'] = 6
#test_data.loc[test_data['RoofMatl'] == 'Roll', 'RoofMatl'] = 7
#test_data.loc[test_data['RoofMatl'] == 'ClyTile', 'RoofMatl'] = 8

test_data.loc[test_data['Exterior1st'] == 'nan', 'Exterior1st'] = 0
test_data.loc[test_data['Exterior1st'] == 'VinylSd', 'Exterior1st'] = 1
test_data.loc[test_data['Exterior1st'] == 'MetalSd', 'Exterior1st'] = 2
test_data.loc[test_data['Exterior1st'] == 'Wd Sdng', 'Exterior1st'] = 3
test_data.loc[test_data['Exterior1st'] == 'HdBoard', 'Exterior1st'] = 4
test_data.loc[test_data['Exterior1st'] == 'BrkFace', 'Exterior1st'] = 5
test_data.loc[test_data['Exterior1st'] == 'WdShing', 'Exterior1st'] = 6
test_data.loc[test_data['Exterior1st'] == 'CemntBd', 'Exterior1st'] = 7
test_data.loc[test_data['Exterior1st'] == 'Plywood', 'Exterior1st'] = 8
test_data.loc[test_data['Exterior1st'] == 'AsbShng', 'Exterior1st'] = 9
test_data.loc[test_data['Exterior1st'] == 'Stucco', 'Exterior1st'] = 10
test_data.loc[test_data['Exterior1st'] == 'BrkComm', 'Exterior1st'] = 11
test_data.loc[test_data['Exterior1st'] == 'AsphShn', 'Exterior1st'] = 12
test_data.loc[test_data['Exterior1st'] == 'Stone', 'Exterior1st'] = 13
test_data.loc[test_data['Exterior1st'] == 'ImStucc', 'Exterior1st'] = 14
test_data.loc[test_data['Exterior1st'] == 'CBlock', 'Exterior1st'] = 15

test_data.loc[test_data['Exterior2nd'] == 'nan', 'Exterior2nd'] = 0
test_data.loc[test_data['Exterior2nd'] == 'VinylSd', 'Exterior2nd'] = 1
test_data.loc[test_data['Exterior2nd'] == 'MetalSd', 'Exterior2nd'] = 2
test_data.loc[test_data['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 3
test_data.loc[test_data['Exterior2nd'] == 'HdBoard', 'Exterior2nd'] = 4
test_data.loc[test_data['Exterior2nd'] == 'Plywood', 'Exterior2nd'] = 5
test_data.loc[test_data['Exterior2nd'] == 'Wd Sdng', 'Exterior2nd'] = 6
test_data.loc[test_data['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 7
test_data.loc[test_data['Exterior2nd'] == 'BrkFace', 'Exterior2nd'] = 8
test_data.loc[test_data['Exterior2nd'] == 'Stucco', 'Exterior2nd'] = 9
test_data.loc[test_data['Exterior2nd'] == 'AsbShng', 'Exterior2nd'] = 10
test_data.loc[test_data['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 11
test_data.loc[test_data['Exterior2nd'] == 'ImStucc', 'Exterior2nd'] = 12
test_data.loc[test_data['Exterior2nd'] == 'AsphShn', 'Exterior2nd'] = 13
test_data.loc[test_data['Exterior2nd'] == 'Stone', 'Exterior2nd'] = 14
test_data.loc[test_data['Exterior2nd'] == 'Other', 'Exterior2nd'] = 15
test_data.loc[test_data['Exterior2nd'] == 'CBlock', 'Exterior2nd'] = 16

test_data.loc[test_data['MasVnrType'] == 'nan', 'MasVnrType'] = 0
test_data.loc[test_data['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 1
test_data.loc[test_data['MasVnrType'] == 'None', 'MasVnrType'] = 2
test_data.loc[test_data['MasVnrType'] == 'Stone', 'MasVnrType'] = 3
test_data.loc[test_data['MasVnrType'] == 'BrkCmn', 'MasVnrType'] = 4

test_data.loc[test_data['ExterQual'] == 'nan', 'ExterQual'] = 0
test_data.loc[test_data['ExterQual'] == 'Gd', 'ExterQual'] = 1
test_data.loc[test_data['ExterQual'] == 'TA', 'ExterQual'] = 2
test_data.loc[test_data['ExterQual'] == 'Ex', 'ExterQual'] = 3
test_data.loc[test_data['ExterQual'] == 'Fa', 'ExterQual'] = 4

test_data.loc[test_data['ExterCond'] == 'nan', 'ExterCond'] = 0
test_data.loc[test_data['ExterCond'] == 'TA', 'ExterCond'] = 1
test_data.loc[test_data['ExterCond'] == 'Gd', 'ExterCond'] = 2
test_data.loc[test_data['ExterCond'] == 'Fa', 'ExterCond'] = 3
test_data.loc[test_data['ExterCond'] == 'Po', 'ExterCond'] = 4
test_data.loc[test_data['ExterCond'] == 'Ex', 'ExterCond'] = 5

test_data.loc[test_data['Foundation'] == 'nan', 'Foundation'] = 0
test_data.loc[test_data['Foundation'] == 'PConc', 'Foundation'] = 1
test_data.loc[test_data['Foundation'] == 'CBlock', 'Foundation'] = 2
test_data.loc[test_data['Foundation'] == 'BrkTil', 'Foundation'] = 3
test_data.loc[test_data['Foundation'] == 'Wood', 'Foundation'] = 4
test_data.loc[test_data['Foundation'] == 'Slab', 'Foundation'] = 5
test_data.loc[test_data['Foundation'] == 'Stone', 'Foundation'] = 6

test_data.loc[test_data['BsmtQual'] == 'nan', 'BsmtQual'] = 0
test_data.loc[test_data['BsmtQual'] == 'Gd', 'BsmtQual'] = 1
test_data.loc[test_data['BsmtQual'] == 'TA', 'BsmtQual'] = 2
test_data.loc[test_data['BsmtQual'] == 'Ex', 'BsmtQual'] = 3
test_data.loc[test_data['BsmtQual'] == 'Fa', 'BsmtQual'] = 4

test_data.loc[test_data['BsmtCond'] == 'nan', 'BsmtCond'] = 0
test_data.loc[test_data['BsmtCond'] == 'TA', 'BsmtCond'] = 1
test_data.loc[test_data['BsmtCond'] == 'Gd', 'BsmtCond'] = 2
test_data.loc[test_data['BsmtCond'] == 'Fa', 'BsmtCond'] = 3
test_data.loc[test_data['BsmtCond'] == 'Po', 'BsmtCond'] = 4

test_data.loc[test_data['BsmtExposure'] == 'nan', 'BsmtExposure'] = 0
test_data.loc[test_data['BsmtExposure'] == 'No', 'BsmtExposure'] = 1
test_data.loc[test_data['BsmtExposure'] == 'Gd', 'BsmtExposure'] = 2
test_data.loc[test_data['BsmtExposure'] == 'Mn', 'BsmtExposure'] = 3
test_data.loc[test_data['BsmtExposure'] == 'Av', 'BsmtExposure'] = 4

test_data.loc[test_data['BsmtFinType1'] == 'nan', 'BsmtFinType1'] = 0
test_data.loc[test_data['BsmtFinType1'] == 'GLQ', 'BsmtFinType1'] = 1
test_data.loc[test_data['BsmtFinType1'] == 'ALQ', 'BsmtFinType1'] = 2
test_data.loc[test_data['BsmtFinType1'] == 'Unf', 'BsmtFinType1'] = 3
test_data.loc[test_data['BsmtFinType1'] == 'Rec', 'BsmtFinType1'] = 4
test_data.loc[test_data['BsmtFinType1'] == 'BLQ', 'BsmtFinType1'] = 5
test_data.loc[test_data['BsmtFinType1'] == 'LwQ', 'BsmtFinType1'] = 6

test_data.loc[test_data['BsmtFinType2'] == 'nan', 'BsmtFinType2'] = 0
test_data.loc[test_data['BsmtFinType2'] == 'GLQ', 'BsmtFinType2'] = 1
test_data.loc[test_data['BsmtFinType2'] == 'ALQ', 'BsmtFinType2'] = 2
test_data.loc[test_data['BsmtFinType2'] == 'Unf', 'BsmtFinType2'] = 3
test_data.loc[test_data['BsmtFinType2'] == 'Rec', 'BsmtFinType2'] = 4
test_data.loc[test_data['BsmtFinType2'] == 'BLQ', 'BsmtFinType2'] = 5
test_data.loc[test_data['BsmtFinType2'] == 'LwQ', 'BsmtFinType2'] = 6

test_data.loc[test_data['Heating'] == 'nan', 'Heating'] = 0
test_data.loc[test_data['Heating'] == 'GasA', 'Heating'] = 1
test_data.loc[test_data['Heating'] == 'GasW', 'Heating'] = 2
test_data.loc[test_data['Heating'] == 'Wall', 'Heating'] = 3
test_data.loc[test_data['Heating'] == 'OthW', 'Heating'] = 4
test_data.loc[test_data['Heating'] == 'Floor', 'Heating'] = 5
test_data.loc[test_data['Heating'] == 'Grav', 'Heating'] = 6

test_data.loc[test_data['HeatingQC'] == 'nan', 'HeatingQC'] = 0
test_data.loc[test_data['HeatingQC'] == 'Ex', 'HeatingQC'] = 1
test_data.loc[test_data['HeatingQC'] == 'Gd', 'HeatingQC'] = 2
test_data.loc[test_data['HeatingQC'] == 'TA', 'HeatingQC'] = 3
test_data.loc[test_data['HeatingQC'] == 'Fa', 'HeatingQC'] = 4
test_data.loc[test_data['HeatingQC'] == 'Po', 'HeatingQC'] = 5

test_data.loc[test_data['CentralAir'] == 'nan', 'CentralAir'] = 0
test_data.loc[test_data['CentralAir'] == 'Y', 'CentralAir'] = 1
test_data.loc[test_data['CentralAir'] == 'N', 'CentralAir'] = 0

test_data.loc[test_data['Electrical'] == 'nan', 'Electrical'] = 0
test_data.loc[test_data['Electrical'] == 'SBrkr', 'Electrical'] = 1
test_data.loc[test_data['Electrical'] == 'FuseF', 'Electrical'] = 2
test_data.loc[test_data['Electrical'] == 'FuseA', 'Electrical'] = 3
test_data.loc[test_data['Electrical'] == 'FuseP', 'Electrical'] = 4
#test_data.loc[test_data['Electrical'] == 'Mix', 'Electrical'] = 5

test_data.loc[test_data['KitchenQual'] == 'nan', 'KitchenQual'] = 0
test_data.loc[test_data['KitchenQual'] == 'Gd', 'KitchenQual'] = 1
test_data.loc[test_data['KitchenQual'] == 'TA', 'KitchenQual'] = 2
test_data.loc[test_data['KitchenQual'] == 'Ex', 'KitchenQual'] = 3
test_data.loc[test_data['KitchenQual'] == 'Fa', 'KitchenQual'] = 4

test_data.loc[test_data['Functional'] == 'nan', 'Functional'] = 0
test_data.loc[test_data['Functional'] == 'Typ', 'Functional'] = 1
test_data.loc[test_data['Functional'] == 'Min1', 'Functional'] = 2
test_data.loc[test_data['Functional'] == 'Maj1', 'Functional'] = 3
test_data.loc[test_data['Functional'] == 'Min2', 'Functional'] = 4
test_data.loc[test_data['Functional'] == 'Mod', 'Functional'] = 5
test_data.loc[test_data['Functional'] == 'Maj2', 'Functional'] = 6
test_data.loc[test_data['Functional'] == 'Sev', 'Functional'] = 7

test_data.loc[test_data['FireplaceQu'] == 'nan', 'FireplaceQu'] = 0
test_data.loc[test_data['FireplaceQu'] == 'TA', 'FireplaceQu'] = 1
test_data.loc[test_data['FireplaceQu'] == 'Gd', 'FireplaceQu'] = 2
test_data.loc[test_data['FireplaceQu'] == 'Fa', 'FireplaceQu'] = 3
test_data.loc[test_data['FireplaceQu'] == 'Ex', 'FireplaceQu'] = 4
test_data.loc[test_data['FireplaceQu'] == 'Po', 'FireplaceQu'] = 5

test_data.loc[test_data['GarageType'] == 'nan', 'GarageType'] = 0
test_data.loc[test_data['GarageType'] == 'Attchd', 'GarageType'] = 1
test_data.loc[test_data['GarageType'] == 'Detchd', 'GarageType'] = 2
test_data.loc[test_data['GarageType'] == 'BuiltIn', 'GarageType'] = 3
test_data.loc[test_data['GarageType'] == 'CarPort', 'GarageType'] = 4
test_data.loc[test_data['GarageType'] == 'Basment', 'GarageType'] = 5
test_data.loc[test_data['GarageType'] == '2Types', 'GarageType'] = 6

test_data.loc[test_data['GarageFinish'] == 'nan', 'GarageFinish'] = 0
test_data.loc[test_data['GarageFinish'] == 'RFn', 'GarageFinish'] = 1
test_data.loc[test_data['GarageFinish'] == 'Unf', 'GarageFinish'] = 2
test_data.loc[test_data['GarageFinish'] == 'Fin', 'GarageFinish'] = 3

test_data.loc[test_data['GarageQual'] == 'nan', 'GarageQual'] = 0
test_data.loc[test_data['GarageQual'] == 'TA', 'GarageQual'] = 1
test_data.loc[test_data['GarageQual'] == 'Fa', 'GarageQual'] = 2
test_data.loc[test_data['GarageQual'] == 'Gd', 'GarageQual'] = 3
test_data.loc[test_data['GarageQual'] == 'Ex', 'GarageQual'] = 4
test_data.loc[test_data['GarageQual'] == 'Po', 'GarageQual'] = 5

test_data.loc[test_data['GarageCond'] == 'nan', 'GarageCond'] = 0
test_data.loc[test_data['GarageCond'] == 'TA', 'GarageCond'] = 1
test_data.loc[test_data['GarageCond'] == 'Fa', 'GarageCond'] = 2
test_data.loc[test_data['GarageCond'] == 'Gd', 'GarageCond'] = 3
test_data.loc[test_data['GarageCond'] == 'Ex', 'GarageCond'] = 4
test_data.loc[test_data['GarageCond'] == 'Po', 'GarageCond'] = 5

test_data.loc[test_data['PavedDrive'] == 'nan', 'PavedDrive'] = 0
test_data.loc[test_data['PavedDrive'] == 'Y', 'PavedDrive'] = 1
test_data.loc[test_data['PavedDrive'] == 'P', 'PavedDrive'] = 2
test_data.loc[test_data['PavedDrive'] == 'N', 'PavedDrive'] = 0

test_data.loc[test_data['PoolQC'] == 'nan', 'PoolQC'] = 0
test_data.loc[test_data['PoolQC'] == 'Ex', 'PoolQC'] = 1
test_data.loc[test_data['PoolQC'] == 'Fa', 'PoolQC'] = 2
test_data.loc[test_data['PoolQC'] == 'Gd', 'PoolQC'] = 3

test_data.loc[test_data['Fence'] == 'nan', 'Fence'] = 0
test_data.loc[test_data['Fence'] == 'MnPrv', 'Fence'] = 1
test_data.loc[test_data['Fence'] == 'GdWo', 'Fence'] = 2
test_data.loc[test_data['Fence'] == 'GdPrv', 'Fence'] = 3
test_data.loc[test_data['Fence'] == 'MnWw', 'Fence'] = 4

test_data.loc[test_data['MiscFeature'] == 'nan', 'MiscFeature'] = 0
test_data.loc[test_data['MiscFeature'] == 'Shed', 'MiscFeature'] = 1
test_data.loc[test_data['MiscFeature'] == 'Gar2', 'MiscFeature'] = 2
test_data.loc[test_data['MiscFeature'] == 'Othr', 'MiscFeature'] = 3
test_data.loc[test_data['MiscFeature'] == 'TenC', 'MiscFeature'] = 4

test_data.loc[test_data['SaleType'] == 'nan', 'SaleType'] = 0
test_data.loc[test_data['SaleType'] == 'WD', 'SaleType'] = 1
test_data.loc[test_data['SaleType'] == 'New', 'SaleType'] = 2
test_data.loc[test_data['SaleType'] == 'COD', 'SaleType'] = 3
test_data.loc[test_data['SaleType'] == 'ConLD', 'SaleType'] = 4
test_data.loc[test_data['SaleType'] == 'ConLI', 'SaleType'] = 5
test_data.loc[test_data['SaleType'] == 'CWD', 'SaleType'] = 6
test_data.loc[test_data['SaleType'] == 'ConLw', 'SaleType'] = 7
test_data.loc[test_data['SaleType'] == 'Con', 'SaleType'] = 8
test_data.loc[test_data['SaleType'] == 'Oth', 'SaleType'] = 9

test_data.loc[test_data['SaleCondition'] == 'nan', 'SaleCondition'] = 0
test_data.loc[test_data['SaleCondition'] == 'Normal', 'SaleCondition'] = 1
test_data.loc[test_data['SaleCondition'] == 'Abnorml', 'SaleCondition'] = 2
test_data.loc[test_data['SaleCondition'] == 'Partial', 'SaleCondition'] = 3
test_data.loc[test_data['SaleCondition'] == 'AdjLand', 'SaleCondition'] = 4
test_data.loc[test_data['SaleCondition'] == 'Alloca', 'SaleCondition'] = 5
test_data.loc[test_data['SaleCondition'] == 'Family', 'SaleCondition'] = 6

test_data_processed = test_data.fillna(0)
test_data_processed = test_data_processed[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']]

result = model.predict(test_data_processed, batch_size=32)
test_data['SalePrice'] = result

submission = test_data[['Id', 'SalePrice']]
submission.to_csv('submit_file.csv', index = False)