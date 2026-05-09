from prophet import Prophet

def train_prophet_model(df):
    
    model =Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
        )
    model.fit(df)

    return model


def predict(model, periods=30, frequency="D"):

    future= model.make_future_dataframe(
        periods=periods,
        freq=frequency,
        include_history=False
    )
    prediction= model.predict(future)

    result= prediction[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return result

def train_and_forecast(df, periods=30):

    model= train_prophet_model(df)
    prediction= predict(
        model=model,
        periods=periods
    )
    return model, prediction
