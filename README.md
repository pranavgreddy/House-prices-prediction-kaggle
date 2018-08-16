# House-prediction-kaggle
House prediction_advanced regression

A house rate prediction challenge with training data of 1460 records and 81 columns and test data of 1459 records

# Procedure:
# Feature engineering
1.I have removed the columns which has more than 80% of missing values in training data
2.After going through the data I have also removed some of the un important columns which I felt not important in predicting the cost of the house
3.One hot encoding of factor varibles
# Missing value imputation
2.As the data contains missing values,I have imputed missing values in numerical columns by median(as it is not affected by outliers) and factor columns with mode(to make sure that no new levels will be added (as it may happen if we use median and mean) in both test and train data.
# Model building
As the data has many columns I have use regularisation techniques and tree techniques
I have tried other techniques but it did not work well.
Ridge:I have used this model to predict the test data for which I got a position of 4488 in leader board(not a good one)

# Dimensionality reduction technique
i have used PCA on it 
1.I tried PCA on data after removinf features which I felt not important(i still removed columns with >80% null values)
got a result of 2226 rank.
2.I tried PCA with out removing features(i still removed columns with >80% null values) got a rank of 2069(yeah its better compared to previous one)

Thank you
