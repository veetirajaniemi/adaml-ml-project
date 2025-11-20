import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=";", dtype="str")
    return df


def preprocess_data(df, start_date="17/12/2006", end_date="26/11/2010"):

    #Combine date and time
    df["datetime"] = pd.to_datetime(df["Date"]+" "+df["Time"], dayfirst=True, errors="coerce")
    #Convert other columns to numeric
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"],errors="coerce")
    df["Global_reactive_power"] = pd.to_numeric(df["Global_reactive_power"],errors="coerce")
    df["Voltage"] = pd.to_numeric(df["Voltage"],errors="coerce")
    df["Global_intensity"] = pd.to_numeric(df["Global_intensity"],errors="coerce")
    df["Sub_metering_1"] = pd.to_numeric(df["Sub_metering_1"],errors="coerce")
    df["Sub_metering_2"] = pd.to_numeric(df["Sub_metering_2"],errors="coerce")
    df["Sub_metering_3"] = pd.to_numeric(df["Sub_metering_3"],errors="coerce")


    #Timestamp rounded to floor hour
    df["timestamp"] = df["datetime"].dt.floor("h")

    # Interpolate missing values for short gaps up to 60 minutes
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df.set_index("datetime",inplace=True)
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=60)

    # Resample to hourly data by taking mean of each hour
    df_hourly = df.resample("h").agg({
        "Global_active_power": "mean",
        "Global_reactive_power": "mean",
        "Global_intensity": "mean",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "mean",
        "Sub_metering_2": "mean",
        "Sub_metering_3": "mean"
    })

    df_hourly = df_hourly[(df_hourly.index >= pd.to_datetime(start_date, dayfirst=True)) & 
                        (df_hourly.index < pd.to_datetime(end_date, dayfirst=True))]

    # Fill remaining missing values using weekly seasonality
    df_hourly = df_hourly.fillna(df_hourly.shift(freq="168h"))
    return df_hourly

#Function for building sequential inputs

def build_sequences(y,n_lags,h=1):
    X_seq = np.zeros((len(y)-n_lags,n_lags//h))
    y_seq = y[(n_lags):]
    for i in range(len(y_seq)):
        X_seq[i,:] = y[i:(i+n_lags):h]
        
    return X_seq, y_seq

#Function for calculating mean arctangent absolute percentage error
def mean_arctangent_absolute_percentage_error(y_actual,y_prediction):
    return np.mean(np.arctan(np.abs((y_actual-y_prediction)/y_actual)))

#Train and validation loop functions for pytorch-models.

def trainloop(dataloader,model,loss_fn,optimizer,batch_size):
    size=len(dataloader.dataset)
    trainloss=0
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        #Prediction and loss
        pred=model(X)
        loss=loss_fn(pred,y.view(-1,1))
        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #Loss for each batch
        loss,current=loss.item(),batch*batch_size+len(X)
        #Add to total loss
        trainloss+=loss
    #print(f"Trainingloss: {trainloss/size}")
    return trainloss/size

def validationloop(dataloader,model,loss_fn,min_validloss):
    size=len(dataloader.dataset)
    model.eval()
    validloss=0
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            #Prediction and loss
            pred=model(X)
            loss=loss_fn(pred,y.view(-1,1))
            validloss+=loss.item()
        #print(f"Validation loss: {validloss/size}")

    return min_validloss, validloss/size