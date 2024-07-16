import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima import model_selection
from pmdarima.arima import auto_arima

def getYFinanceData(ticker, period, interval):
  tickerObj = yf.Ticker(ticker)
  allData = tickerObj.history(period=period, interval=interval)
  #allData = tickerObj.history(period=period, proxy = "PROXY_SERVER")
  return allData
def cleanYFinanceData(allData):
  allData.index = allData.index.tz_convert('America/New_York')
  rate = pd.DataFrame(allData["Close"])
  rate = rate.reset_index()
  rate = rate.rename(columns={'Datetime': 'Date'})
  rate["Date"] = rate["Date"].apply(lambda t: t.strftime('%Y-%m-%d %H:%M:%S'))
  rate["Date"] = pd.to_datetime(rate["Date"])
  rate.set_index("Date", inplace=True)

  return rate

def imputeTimeSeries(rate):
    if (rate["Close"].isna().sum() == 0):
        return rate

    else:
        rate["Close"] = rate["Close"].interpolate(method='linear', inplace=True, limit_direction="both")
        return rate
def isStationary(rate):
    if (rate.columns.str.contains("Difference").sum() == 0):
        rate["Difference"] = rate["Close"]

    results = adfuller(rate["Difference"])

    # check p-value is less than 0.05
    if results[1] <= 0.05:
        return True;

    else:
        return False;

def isStationaryExplain(rate):
    if (rate.columns.str.contains("Difference").sum() == 0):
        rate["Difference"] = rate["Close"]

    results = adfuller(rate["Difference"])
    resultsLabels = ['Test Statistic for ADF', 'p-value', 'Number of Lagged Observations Used',
                     'Number of Observations Used']

    for value, label in zip(results, resultsLabels):
        print(label + ' : ' + str(value))

    # check p-value is less than 0.05
    if results[1] <= 0.05:
        print(
            "According to the ADF test, the p-value is less than 0.05. Therefore we reject the null hypothesis. We conclude the data is stationary")

    else:
        print(
            "According to the ADF test, the p-value is greater than 0.05. Therefore, we fail to reject the null hypothesis. We conclude the data is non-stationary ")


def difference(rate, degree):
    testRate = rate.copy()

    if (testRate.columns.str.contains("Difference").sum() == 0):
        testRate["Difference"] = testRate["Close"]

    for i in range(degree):
        testRate["Difference"] = testRate["Difference"] - testRate["Difference"].shift(1)

    testRate = testRate.dropna()
    return testRate


def degreeOfTransformation(rate):
  testRate = rate.copy()
  n = 0
  while (n < 5):
    #isStationaryExplain(testRate)
    #print(testRate)

    if (isStationary(testRate)):
      return n

    else:
      testRate = difference(testRate, 1)
      n+=1

  return n


def transformToStationary(rate):
    degree = degreeOfTransformation(rate)
    # print(degree)

    rate = difference(rate, degree)

    rate = rate.drop("Close", axis=1)
    rate = rate.rename(columns={"Difference": "Close"})

    return rate

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.base.tsa_model import ValueWarning
import warnings
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from sklearn.model_selection import train_test_split


# Splitting data into test and training
def getStartTrainData(rate, size=0.5):
    train, test = train_test_split(rate, shuffle=False, stratify=None, test_size=1 - size)
    return train


def getStartTestData(rate, size=0.5):
    train, test = train_test_split(rate, shuffle=False, stratify=None, test_size=size)
    return test


# Transfers the top datapoint from the testing dataset to become the bottom value in the training dataset
def transferTrainTestData(train, test):
    if len(test) == 0:
        return train, test

    datapoint = test.iloc[[0]]
    train = train.append(datapoint)
    test = test.iloc[1:]
    return train, test
def fitARIMAManual(rate, pTerm, degree, qTerm, train):
  degree = degreeOfTransformation(rate)

  arimaObj = sm.tsa.arima.ARIMA(train['Close'], order=(pTerm, degree, qTerm))

  modelARIMAManual = arimaObj.fit()
  return modelARIMAManual

def getMaxForesight(modelARIMAManual):
  maxForesight = max(int(modelARIMAManual.model_orders["ar"]), int(modelARIMAManual.model_orders["ma"]))
  return maxForesight

def predictARIMAManual(modelARIMAManual, train):
  predictionARIMAManual = modelARIMAManual.predict(start = len(train), end = len(train) - 1 + getMaxForesight(modelARIMAManual), typ='levels',dynamic = True)
  #predictionARIMAManual = modelARIMAManual.forecast(steps = 1)
  return predictionARIMAManual


def getARIMAParameters(rateAll):
    autoARIMA = auto_arima(rateAll, start_p=0, start_q=0, max_p=10, max_q=10, m=0,
                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)  # set to stepwise

    pTerm = autoARIMA.order[0]
    degree = autoARIMA.order[1]
    qTerm = autoARIMA.order[2]

    params = []

    params.append(pTerm)
    params.append(degree)
    params.append(qTerm)

    return params


from datetime import datetime as dt


def justify(a, invalid_val=0, axis=1, side='left'):
    if invalid_val is np.nan:
        mask = ~np.isnan(a)
    else:
        mask = a!=invalid_val
    justified_mask = np.sort(mask,axis=axis)
    if (side=='up') | (side=='left'):
        justified_mask = np.flip(justified_mask,axis=axis)
    out = np.full(a.shape, invalid_val)
    if axis==1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out

def cleanAllARIMAPredictions(predictions, modelARIMAManual):
    foresight = getMaxForesight(modelARIMAManual)

    predictions = predictions.T
    predictions = pd.DataFrame(justify(predictions.to_numpy(), invalid_val=np.nan),
                                   index=predictions.index,
                                   columns=predictions.columns)
    predictions = predictions.iloc[:, foresight - 1:foresight]
    predictions = predictions.rename(columns={"predicted_mean": "Predicted Close With Lag " + str(foresight)})

    predictions.reset_index(inplace=True)
    predictions['index'] = predictions['index'].apply(lambda x: str(x))
    predictions['index'] = predictions['index'].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))

    predictions.index = pd.to_datetime(predictions.index, format="%Y-%m-%d")

    predictions = predictions.rename(columns={predictions.columns[0]: "Date"})
    predictions.set_index("Date", inplace=True)

    predictions = predictions.dropna();

    return predictions

def allDynamicARIMAPredictions(rate, modelARIMAManualStarting, trainStarting, testStarting):

    trainPredict = trainStarting.copy()
    testPredict = testStarting.copy()
    predictions = pd.DataFrame()

    modelARIMAManualPredict = modelARIMAManualStarting
    degree = degreeOfTransformation(rate)
    pTerm = modelARIMAManualPredict.model_orders["ar"]
    qTerm = modelARIMAManualPredict.model_orders["ma"]

    foresight = getMaxForesight(modelARIMAManualStarting)

    while (True):
        modelARIMAManualPredict = fitARIMAManual(rate, pTerm, degree, qTerm, trainPredict)
        prediction = predictARIMAManual(modelARIMAManualPredict, trainPredict)

        if (foresight == 1):
            prediction = pd.DataFrame(prediction, columns=['Predicted Close'])
            predictions = predictions.append(prediction)

        else:
            a = prediction.T
            predictions = predictions.append(a)

        if len(testPredict) == 0:
            break

        trainPredict, testPredict = transferTrainTestData(trainPredict, testPredict)

            # print("Train length: ", len(trainPredict))
            # print("Test length: ", len(testPredict))
        trainPredict = pd.DataFrame(trainPredict)
        testPredict = pd.DataFrame(testPredict)

    if foresight == 1:
        predictions.index = pd.to_datetime(predictions.index, format="%Y-%m-%d")

    else:
        predictions = cleanAllARIMAPredictions(predictions, modelARIMAManualStarting)

    predictions.index.rename('Date', inplace=True)

    return predictions

    # *
def add(df):
    next = df.index[-1] + pd.Timedelta(minutes=1)
    new = pd.Series([None], index=df.columns, name=next)
    df = df.append(new)
    return df

def mergePredictionsAndTest(predictions, rate):
    testData= getStartTestData(rate)
    testData = add(testData)
    predictionsAndTest = testData.merge(predictions, left_index=True, right_index=True)
    return predictionsAndTest

def run():
    asset = "GOOG"
    allData = getYFinanceData(asset, "1d", '1m')
    allData = allData.iloc[:-1, :]

    rate = cleanYFinanceData(allData)
    rate = imputeTimeSeries(rate)
    rate1 = rate

    # TEMPORARY FIX TO WEEKEND BUG (shows price of ASSET on weekends)
    # THIS REMOVES THE BUGGED DATAPOINT
    # allData = allData.iloc[:-1 , :]

    # MAIN DRIVER CODE pt.2
    # **DO NOT CHANGE PERIOD FROM MAX, use the first instance of getYFinanceData() to change period of stock
    allDataMax = getYFinanceData(asset, "7d", "1m")
    rateAll = cleanYFinanceData(allDataMax)
    rateAll = imputeTimeSeries(rateAll)

    # MAIN DRIVER CODE
    parameters = getARIMAParameters(rateAll)

    if (parameters[0] + parameters[2]) < 1:
        raise Exception(
            "There are no autoregressive or moving average components to this data. This data is unsuitable for ARIMA analysis. Pick another dataset.")

    modelARIMAManual = fitARIMAManual(rate, parameters[0], parameters[1], parameters[2], getStartTrainData(rate))

    predictions = allDynamicARIMAPredictions(rate, modelARIMAManual, getStartTrainData(rate), getStartTestData(rate))
    predictionsAndTest = mergePredictionsAndTest(predictions, rate1)
    print(predictionsAndTest['Predicted Close'].iloc[-1])
    return predictionsAndTest['Predicted Close'].iloc[-1]


run()