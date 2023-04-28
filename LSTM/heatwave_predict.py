import pandas as pd
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def process_temperature_data(realTemp, realTemp_, df):
    # create a new DataFrame with the temperature data
    temp_ = pd.DataFrame(realTemp_, index=df.index[6:11542], columns=["temp"])

    # convert the index to datetime format and set it as the new index
    temp_.index = pd.to_datetime(temp_.index, dayfirst=True)
    temp_.index = temp_.index.date

    # round the temperature values to 2 decimal places
    temp_ = temp_["temp"].round(2)

    # create another DataFrame for the remaining temperature data
    temp__ = pd.DataFrame(realTemp, index=df.index[11548:], columns=["temp"])
    temp__.index = pd.to_datetime(temp__.index, dayfirst=True)
    temp__.index = temp__.index.date
    temp__ = temp__["temp"].round(2)

    return temp_, temp__


def calculate_normal_temperatures(realTemp, realTemp_, df):
    temp_, temp__ = process_temperature_data(realTemp, realTemp_, df)

    # create an empty Series to hold the mean temperatures
    mean_temps = pd.Series(index=temp__.index)

    # loop through each date in the DataFrame
    for date in temp__.index:
        # extract the month and day for the current date
        month, day = date.month, date.day

        # check if the current date is a leap day and whether it exists in the previous year
        if (
            month == 2
            and day == 29
            and not pd.Timestamp(f"{date.year - 1}-01-01").is_leap_year
        ):
            # skip leap days if they do not exist in the previous year
            continue
        if month == 2 and day == 29:
            try:
                prev_date = pd.Timestamp(f"{date.year - 1}-{month:02}-{day:02}")
                prev_date = prev_date.date()
                temp = temp_.loc[prev_date]
            except KeyError:
                temp = None
        else:
            # loop over the previous 30 years
            mean_temp_sum = 0
            count = 0  # to keep track of the number of years we have valid temperature data for
            for year_offset in range(1, 31):
                # calculate the year for the previous year offset
                year = date.year - year_offset

                # create a new date for the same month and day in the previous year offset
                if (
                    month == 2
                    and day == 29
                    and not pd.Timestamp(f"{year}-02-29").is_leap_year
                ):
                    # skip leap years if the current date is a leap day and does not exist in the previous year
                    continue
                try:
                    prev_date = pd.Timestamp(f"{year}-{month:02}-{day:02}")
                    prev_date = prev_date.date()

                    # select the temperature for the previous year offset on the same month and day
                    temp = temp_.loc[prev_date]
                except KeyError:
                    temp = None

                if not pd.isna(temp):
                    # only include valid temperature data in the mean calculation
                    mean_temp_sum += temp
                    count += 1

            if count > 0:
                # calculate the mean temperature for the previous 30 years on that specific date
                mean_temp = mean_temp_sum / count
            else:
                mean_temp = None

        # add the mean temperature to the Series for the current date
        mean_temps[date] = mean_temp + 5

    return mean_temps

def calculate_heatwave_accuracy(real_temp, predict_temp, normal_temp):
    """Calculate the heatwave accuracy percentage based on real and predicted temperatures and normal temperatures."""
    y_true = [1 if temp >= normal_temp[i] else 0 for i,temp in enumerate(real_temp)]
    y_pred = [1 if temp >= normal_temp[i] else 0 for i,temp in enumerate(predict_temp)]
    conf_mat = confusion_matrix(y_true, y_pred)
    accuracy = (conf_mat[0][0] + conf_mat[1][1]) / len(real_temp)
    sns.heatmap(conf_mat, annot=True, cmap='PiYG', fmt='g', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return accuracy

def predict_heatwave(predictTemp, realTemp, realTemp_, df):
    normalTemp = calculate_normal_temperatures(realTemp, realTemp_, df)
    heatWaveReal, heatWavePredict = 0, 0
    for itr in range(normalTemp.shape[0]):
        if predictTemp[itr] >= normalTemp[itr]:
            heatWavePredict += 1
        if realTemp[itr] >= normalTemp[itr]:
            heatWaveReal += 1

    accuracy = calculate_heatwave_accuracy(realTemp, predictTemp, normalTemp)

    return heatWaveReal, heatWavePredict, accuracy