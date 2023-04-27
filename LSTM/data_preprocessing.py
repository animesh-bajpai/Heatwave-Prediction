import pandas as pd

def preprocess_data(filename):
    # Load the data from the CSV file
    df = pd.read_csv(filename, skiprows=15)

    # Convert date columns to string format
    df['YEAR'] = df.YEAR.astype(str)
    df['MO'] = df.MO.astype(str)
    df['DY'] = df.DY.astype(str)

    # Combine date columns to create a datetime column
    df['date'] = df['DY'].str.cat(df['MO'], sep='/')
    df['dateTime'] = df['date'].str.cat(df['YEAR'], sep='/')

    # Drop unnecessary columns and set datetime column as index
    df.drop(['YEAR', 'MO', 'DY', 'date'], axis=1, inplace=True)
    df.set_index('dateTime', inplace=True)

    # Convert index to datetime format
    df.index = pd.to_datetime(df.index, dayfirst=True)

    # Rename columns for easier interpretation
    df.rename(columns={'T2M_MAX': 'tempMax', 'T2M': 'temp', 'T2M_MIN': 'tempMin', 'TS': 'earthSkin', 
                       'RH2M': 'relativeHumidity', 'PS': 'pressure', 'T2MDEW': 'dew'}, inplace=True)

    return df