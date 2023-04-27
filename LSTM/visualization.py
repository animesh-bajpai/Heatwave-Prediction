import matplotlib.pyplot as plt

def plot_train_test_predictions(y_real, y_predict, date_test):
    '''
    Plots the actual and predicted values of training and testing sets in the same plot.

    Parameters:
    y_train (numpy.ndarray): Actual values of the training set.
    y_test (numpy.ndarray): Actual values of the testing set.
    y_pred_train (numpy.ndarray): Predicted values of the training set.
    y_pred_test (numpy.ndarray): Predicted values of the testing set.
    date_test (pandas.core.indexes.datetimes.DatetimeIndex): Dates of the testing set.
    '''
    plt.figure(figsize=(16, 8))
    plt.plot(date_test, y_real, label='Actual Test Data')
    plt.plot(date_test, y_predict, label='Predicted Test Data')
    plt.legend()
    plt.show()

def plot_loss(history):
    '''
    Plots the loss vs epoch graph of the training process.

    Parameters:
    history (keras.callbacks.History): The history object returned by keras.model.fit() method.
    '''
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()