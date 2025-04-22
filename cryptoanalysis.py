import yfinance as yf
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Function to get data from Yahoo Finance
def get_yahoo_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    
    if data.empty:
        st.error("No data found for the given date range. Please try a different range.")
        return None
    
    data.reset_index(inplace=True)  # Ensure 'Date' is a column, not an index
    data.rename(columns={"Date": "date", "Close": "price"}, inplace=True)

    return data[['date', 'price']].dropna()  # Keep only relevant columns

# Function to get data from CoinGecko
def get_coingecko_data(crypto_id, days):
    cg = CoinGeckoAPI()
    prices = cg.get_coin_market_chart_by_id(id=crypto_id, vs_currency='usd', days=days)
    
    if not prices or 'prices' not in prices:
        st.error("Failed to retrieve data from CoinGecko.")
        return None
    
    df = pd.DataFrame(prices['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop(columns=['timestamp'], inplace=True)
    
    return df[['date', 'price']].dropna()

# Function to calculate volatility
def calculate_volatility(data):
    data = data.copy()
    data['Returns'] = data['price'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    return data

# Function to forecast prices using ARIMA
def forecast_arima(data, steps=50, order=(15,1,0)):
    try:
        model = ARIMA(data['price'].dropna(), order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        st.error(f"ARIMA Forecasting Error: {e}")
        return None

# Function to plot price and volatility
def plot_data(data, log_scale=False):
    if data is None or data.empty:
        st.error("No data available for plotting.")
        return

    fig = go.Figure()

    # Price line (Primary Y-axis)
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['price'], 
        mode='lines', 
        name='Price', 
        line=dict(color='blue')
    ))

    # Volatility line (Secondary Y-axis)
    if 'Volatility' in data.columns and data['Volatility'].notna().any():
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['Volatility'],  
            mode='lines', 
            name='Volatility', 
            line=dict(color='red'),
            yaxis="y2"
        ))

    # Layout update with properly linked secondary Y-axis
    fig.update_layout(
        title="Price & Volatility Analysis",
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Price (Log Scale)" if log_scale else "Price",
            type="log" if log_scale else "linear",
            side="left"
        ),  
        yaxis2=dict(
            title="Volatility",
            overlaying="y",
            side="right",
            showgrid=False,
            anchor="x"
        )
    )

    st.plotly_chart(fig)

# Streamlit App
def main():
    st.title('Cryptocurrency Time Series Analysis')
    
    option = st.selectbox("Select Data Source", ["Yahoo Finance", "CoinGecko"])
    
    log_scale = st.checkbox("Use Log Scale for Price", value=False)

    if option == "Yahoo Finance":
        symbol = st.text_input("Enter Yahoo Finance Symbol (e.g., BTC-USD)", "BTC-USD")
        start = st.date_input("Start Date")
        end = st.date_input("End Date")
        
        if st.button("Fetch Data"):
            data = get_yahoo_data(symbol, start, end)
            
            if data is not None and not data.empty:
                data = calculate_volatility(data)
                st.write("### Data Preview", data.tail())
                plot_data(data, log_scale)
                
                # ARIMA Forecasting
                forecast = forecast_arima(data)
                if forecast is not None:
                    st.write("### ARIMA Forecast for Next 30 Days")
                    st.line_chart(forecast)
    
    elif option == "CoinGecko":
        crypto_id = st.text_input("Enter CoinGecko ID (e.g., bitcoin)", "bitcoin")
        days = st.slider("Number of Days", 30, 365, 90)
        
        if st.button("Fetch Data"):
            data = get_coingecko_data(crypto_id, days)
            
            if data is not None and not data.empty:
                data = calculate_volatility(data)
                st.write("### Data Preview", data.tail())
                plot_data(data, log_scale)
                
                # ARIMA Forecasting
                forecast = forecast_arima(data)
                if forecast is not None:
                    st.write("### ARIMA Forecast for Next 30 Days")
                    st.line_chart(forecast)

if __name__ == '__main__':
    main()