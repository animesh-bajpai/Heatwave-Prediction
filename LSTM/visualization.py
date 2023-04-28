import matplotlib.pyplot as plt

def plot_train_test_predictions(y_real, y_predict, date_test):
    '''
    Plots the actual and predicted values of actual and predicted sets in the same plot.

    Parameters:
    y_real (numpy.ndarray): Actual values of the data set.
    y_predict (numpy.ndarray): Predicted values of the data set.
    date_test (pandas.core.indexes.datetimes.DatetimeIndex): Dates of the data set.
    '''

    plt.figure(figsize=(16, 8))
    plt.plot(date_test, y_real, color='m', label='Actual Test Data')
    plt.plot(date_test, y_predict, color='k', label='Predicted Test Data')
    plt.legend()
    plt.show()

def plot_loss(history):
    '''
    Plots the loss vs epoch graph of the training process.

    Parameters:
    history (keras.callbacks.History): The history object returned by keras.model.fit() method.
    '''
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['loss'], color='m', label='Train Loss')
    plt.plot(history.history['val_loss'], color='k', label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()