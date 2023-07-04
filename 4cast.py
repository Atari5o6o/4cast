import numpy as np
from yahoo_fin.stock_info import get_data
from get_all_tickers import get_tickers_filtered
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from trade_stat_logger.logger import SimpleLogger


# functiom to get formatted data for specific ticker
def get_ml_formatted_data(ticker, start_date='01/01/2020', end_date='01/01/2022'):
    
    
    #Getting the data from yahoo finance
    df= get_data(ticker=ticker, start_date=start_date, end_date=end_date)

    #Prepare data
    del df['ticker']
    del df['adjclose']
    del df['volume']    

    #we use pct_change(), which calculates the percentage change from the previous row 
    x = df.pct_change()
    x = x.iloc[1:, :] #remove the first row, which is NaN since it did not have a previous row to calculate the percentage change
    df = df.iloc[1:,] #.iloc() is a index based selection method, iloc[1:, :] means all rows starting from index 1 to the end, and all columns
    x.reset_index(drop=True, inplace=True) #reset_index() is used to reset the index of the Data Frame, drop=True means it will delete the old index column

    def cust_filter(row):
        if (row.name +1) % 6 == 0: #sees if selected row is divisible by 6
            return int(row['open'] > 0)#if it is, it will return 1 if the open price is greater than 0, else it will return 0
        else:
            return float('NaN')#if it is not every 6th row, it wil return NaN
    
    x['opened_up'] = x.apply(cust_filter, axis=1)#apply() is used to apply cust_filter() to every row in the data frame (x) to create "opened_up" column
    labels = x['opened_up'].dropna().values.astype(int)#dropna() is used to drop NaN values, the values that return 0 or 1 after cust_filter() is converted to integer using astype(int)
    del x['opened_up']#deletes the "opened_up" column from x as it was only used to create the labels

    # flatten() makes it so we can put it through our model
    chunks = [x.iloc[a:a+5].values.flatten() for a in range(0, len(x), 6)] #"x" is iterated with incremenets of 6, and the 5 consecutive rows of each iteration are "flattened" to one chunk of data

     # sometimes there is an extra chunk of a few rows at the end that we don't want
    if len(chunks) > len(labels):
        chunks.pop(-1)
    data = np.asarray(chunks)#converts chunks list to NumPy array for further preprocessing 


    close_values = df['close'].iloc[4::6].values#iloc[4::6] means every 6th row starting from index 4
    open_values = df['open'].iloc[5::6].values#iloc[5::6] means every 6th row starting from index 5
    return data, labels, close_values, open_values


#function to get formatted data for multiple tickers
def get_ml_data_multiple_tickers(tickers_list, start_date='01/01/2020', end_date='01/01/2022'):
    data_list = []
    labels_list = []
    close_values_list = []
    open_values_list = []
    for ticker in tickers_list:
        data, labels, close_values, open_values = get_ml_formatted_data(ticker, start_date=start_date, end_date=end_date)
        data_list.extend(data)
        labels_list.append(labels)
        close_values_list.append(close_values)
        open_values_list.append(open_values)

    return np.asarray(data_list),np.concatenate(labels_list, axis = None), np.concatenate(close_values_list, axis = None), np.concatenate(open_values_list, axis = None)

tickers = get_tickers_filtered(mktcap_min=200000)#gets all tickers with market cap greater than 200000

data, labels, _,_ = get_ml_data_multiple_tickers(tickers, start_date='01/01/2020', end_date='01/01/2022')#gets data for all tickers

#training the model

model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(8,8,8), max_iter=10000, activation='tanh', random_state=1)
#lbfgs converges fast for small datasets
#hidden layers are essentially the neurons between the input and output layers, in this case there are three layers with 8 neurons each
#activations using tanh functuion prevents neuron deths, neuron deahs cause all outputs to be 0
#setting random_state to 1 (or basically any integer) makes it a deterministic model

X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size=.5)#splits data into training and testing data
model.fit(X_train, y_train)#fits the model to the training data

predicted  = model.predict(X_test)#predicts the labels for the testing data

#evaluating the model
print(metrics.classification_report(y_test, predicted))
print("Accuracy:", model.score(X_test, y_test))#prints the accuracy of the model


model.fit(data, labels)

# STEP 5: Develop trading algorithm
# we could try to match each data point with the corresponding ticker
# however, since we log buys and sells instantly it doesn't matter
ticker = 'ANON'

logger = SimpleLogger()

data, labels, close_vals, open_vals = get_ml_data_multiple_tickers(tickers, start_date='01/10/2019', end_date='05/01/2020')

def get_prediction_values(data):
    # need to encapsulate 1D data in a list b/c sklearn models 
    # don't take single input values
    value = model.predict([data])[0]
    certainty = np.amax(model.predict_proba([data])[0])
    return value, certainty

for i in range(len(labels)):
    value, certainty = get_prediction_values(data[i])
    if certainty > .95:
        if value:
            logger.log(security=ticker, share_price=close_vals[i], shares=100)
            logger.log(security=ticker, share_price=open_vals[i], shares=-100)
        else:
            logger.log(security=ticker, share_price=close_vals[i], shares=-100)
            logger.log(security=ticker, share_price=open_vals[i], shares=100)

# STEP 6: Evaluate Algorithm
logger.graph_statistics()
# print(type(logger))