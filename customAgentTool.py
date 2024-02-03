import os 
import yfinance as yf
from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_openai_tools_agent
from fastapi import FastAPI
from datetime import datetime, timedelta
from typing import List
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

app = FastAPI()  

# Utility functions
def get_stock_price_from_yf(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'].iloc[0], 2)

def get_price_change_percent(symbol, days_ago):
    ticker = yf.Ticker(symbol)

    # Get today's date
    end_date = datetime.now()

    # Get the date N days ago
    start_date = end_date - timedelta(days=days_ago)

    # Convert dates to string format that yfinance can accept
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Get the historical data
    historical_data = ticker.history(start=start_date, end=end_date)

    # Get the closing price N days ago and today's closing price
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]

    # Calculate the percentage change
    percent_change = ((new_price - old_price) / old_price) * 100

    return round(percent_change, 2)

def calculate_performance(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    historical_data = ticker.history(start=start_date, end=end_date)
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]
    percent_change = ((new_price - old_price) / old_price) * 100
    return round(percent_change, 2)

def get_best_performing(stocks, days_ago):
    best_stock = None
    best_performance = None
    for stock in stocks:
        try:
            performance = calculate_performance(stock, days_ago)
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            print(f"Could not calculate performance for {stock}: {e}")
    return best_stock, best_performance

# Tool 1 - uses OpenAI's function calling:
# Describes the input
class StockPriceCheckInput(BaseModel):
    """Input for Stock price check."""

    ticker: str = Field(..., description="Ticker symbol for stock or index")

@tool("StockPriceTool", args_schema=StockPriceCheckInput)
def getStockPriceByTicker(ticker: str) -> str:
    """Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"""

    stockPrice = get_stock_price_from_yf(ticker)
    return stockPrice

# Tool 2
class StockPricePercentageChangeInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")
    days_ago: int = Field(..., description="Int number of days to look back")

@tool("Stock-price-percentage-change", args_schema=StockPricePercentageChangeInput)
def getStockPricePercentageChange(stockticker: str, days_ago: int) -> str:
    """Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"""

    price_change_response = get_price_change_percent(stockticker, days_ago)
    return price_change_response

# Tool 3
class BestPerformingStockInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days_ago: int = Field(..., description="Int number of days to look back")

@tool("Best-performing-stock", args_schema=BestPerformingStockInput)
def getStockPriceStockWithPercentage(stocktickers: List[str], days_ago: int) -> str:
    """Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"""

    price_change_response = get_best_performing(stocktickers, days_ago)
    return price_change_response

# Tool 4 
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [
    getStockPriceByTicker, 
    getStockPricePercentageChange, 
    getStockPriceStockWithPercentage,
    repl_tool
]

# Agent
def get_response(ticker: str):
    llm = ChatOpenAI(temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(
        llm, 
        tools,
        prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": ticker})
    return response['output']

@app.get("/get-stock-price", description="Getting stock price based on the ticker")
async def getPrice(query: str):
    response = get_response(query)
    return { "response": response }