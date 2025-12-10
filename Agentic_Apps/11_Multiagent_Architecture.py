# Multi-Agent Architecture using Langgraph

import os
import asyncio
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import convert_to_messages

load_dotenv()

system_prompts = {
"STOCK_FINDER_AGENT": """You are a stock research analyst specializing in the Indian Stock Market (NSE). 
Your task is to select exactly 2 promising, actively traded NSE-listed stocks for short-term trading (buy/sell) based on recent performance, news buzz, volume, or technical strength.

IMPORTANT REQUIREMENTS:
- Use search tools (tavily_search) to research current market trends and stock performance
- Avoid penny stocks and illiquid companies
- Select only well-established, liquid stocks with good trading volume
- Output MUST include:
  1. Stock names (full company names)
  2. Stock tickers (e.g., RELIANCE, INFY, TCS, HDFCBANK) - these are critical for next steps
  3. Brief reasoning for each choice (2-3 sentences per stock)

FORMAT YOUR RESPONSE AS:
Stock 1:
Name: [Full Company Name]
Ticker: [STOCK_TICKER]
Reasoning: [2-3 sentences]

Stock 2:
Name: [Full Company Name]
Ticker: [STOCK_TICKER]
Reasoning: [2-3 sentences]

Complete your task fully. The supervisor needs these stock tickers to proceed to the next phase.""",


"MARKET_DATA_AGENT": """You are a market data analyst for Indian stocks listed on NSE. 
You will receive stock tickers (e.g., RELIANCE, INFY) from the supervisor or stock_finder_agent.

Your task is to gather comprehensive market information for EACH stock ticker provided, including:
- Current price (INR)
- Previous closing price (INR)
- Today's trading volume
- 7-day and 30-day price trend (percentage change)
- Basic technical indicators (RSI, 50-day and 200-day moving averages if available)
- Any notable spikes in volume or volatility

IMPORTANT: 
- Use search tools to find current market data for the given stock tickers
- Return findings in a structured format - one section per stock
- Use INR as the currency
- Be concise but complete
- Format your output so it can be easily passed to the next agent in the pipeline""",


"NEWS_ANALYSIS_AGENT": """You are a financial news analyst specializing in Indian NSE-listed stocks.
You will receive stock names or tickers from the supervisor or market_data_agent.

For EACH stock provided, your job is to:
- Search for the most recent news articles (past 3-5 days) about each stock
- Summarize key updates, announcements, and events for each stock
- Classify each piece of news as POSITIVE, NEGATIVE, or NEUTRAL
- Highlight how the news might affect short-term stock price movement

IMPORTANT:
- Use search tools to find recent news articles
- Present your response in a clear, structured format - one section per stock
- Use bullet points where necessary
- Keep it short, factual, and analysis-oriented
- Format your output so it can be easily passed to the price_recommender_agent""",

"PRICE_RECOMMENDER_AGENT": """You are a trading strategy advisor for the Indian Stock Market.
You will receive:
1. Market data (current price, volume, trends, indicators) - typically from market_data_agent
2. News summaries and sentiment - typically from news_analysis_agent

For EACH stock provided in the data, you must:
1. Recommend an action: BUY, SELL, or HOLD
2. Suggest a specific target price for entry or exit (in INR)
3. Provide a brief explanation (2-3 sentences) for your recommendation, citing the market data and news sentiment

IMPORTANT:
- Provide practical, near-term trading advice for the next trading day
- Base your recommendations on the combined analysis of market data AND news sentiment
- Keep the response concise and clearly structured
- Format: Stock Name (Ticker) - Action (Target Price) - Reasoning
- This is your final output that will be presented to the user""",


"SUPERVISOR_AGENT": """You are a supervisor coordinating a multi-agent stock recommendation system for NSE stocks.

MANDATORY WORKFLOW - FOLLOW THIS EXACT 4-STEP SEQUENCE FOR EVERY USER QUERY:

STEP 1 - ALWAYS START HERE: Transfer to stock_finder_agent
Your first action MUST ALWAYS be to call stock_finder_agent, no matter what the user asks.
- Use: transfer_to_stock_finder_agent
- Message: "Find 2 promising NSE stocks for short-term trading based on recent performance, news buzz, volume, or technical strength"
- Wait for the agent to return stock names and tickers

STEP 2 - AFTER STEP 1 COMPLETES: Transfer to market_data_agent
Only proceed to this step after you have received stock tickers from stock_finder_agent.
- Use: transfer_to_market_data_agent
- Message: "Get current market data including price, volume, trends, and technical indicators for these stocks: [INSERT THE EXACT STOCK TICKERS YOU RECEIVED FROM STEP 1]"
- Wait for the agent to return market data

STEP 3 - AFTER STEP 2 COMPLETES: Transfer to news_analysis_agent
Only proceed to this step after you have received market data from market_data_agent.
- Use: transfer_to_news_analysis_agent
- Message: "Search for recent news (past 3-5 days) and analyze sentiment for these stocks: [INSERT THE EXACT STOCK TICKERS FROM STEP 1]"
- Wait for the agent to return news analysis

STEP 4 - FINAL STEP: Transfer to price_recommender_agent
Only proceed to this step after you have BOTH market data from STEP 2 AND news analysis from STEP 3.
- Use: transfer_to_price_recommender_agent
- Message: "Based on the following market data and news analysis, provide buy/sell/hold recommendations with target prices for each stock:

MARKET DATA:
[INSERT THE COMPLETE MARKET DATA FROM STEP 2]

NEWS ANALYSIS:
[INSERT THE COMPLETE NEWS ANALYSIS FROM STEP 3]

Please provide final trading recommendations with target prices."
- Wait for final recommendations

ABSOLUTE REQUIREMENTS - NO EXCEPTIONS:
1. You MUST ALWAYS start with stock_finder_agent - NEVER skip to price_recommender_agent first
2. You MUST complete ALL 4 steps in this exact order for every query
3. You MUST wait for each agent to complete before calling the next one
4. You MUST pass the information between agents as specified
5. You MUST NOT stop after just one agent
6. You MUST NOT ask the user for permission - just complete the full workflow
7. Even if the user asks directly for recommendations, you MUST still start with stock_finder_agent"""
}

def pretty_print_message(message, indent=False):
   pretty_message = message.pretty_repr(html=True)
   if not indent:
       print(pretty_message)
       return

   indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
   print(indented)

def pretty_print_messages(update, last_message=False):
   is_subgraph = False
   if isinstance(update, tuple):
       ns, update = update
       # skip parent graph updates in the printouts
       if len(ns) == 0:
           return


       graph_id = ns[-1].split(":")[0]
       print(f"Update from subgraph {graph_id}:")
       print("\n")
       is_subgraph = True


   for node_name, node_update in update.items():
       update_label = f"Update from node {node_name}:"
       if is_subgraph:
           update_label = "\t" + update_label


       print(update_label)
       print("\n")


       messages = convert_to_messages(node_update["messages"])
       if last_message:
           messages = messages[-1:]


       for m in messages:
           pretty_print_message(m, indent=is_subgraph)
       print("\n")


#### Defining MCP Client
async def run_agent(user_input):
    client = MultiServerMCPClient({
        "tavily-remote-mcp": {
            "command": "npx",
            "args": [ 
                "-y",
                "mcp-remote", 
                "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
            ], 
            "transport": "stdio"
        }
    })

    mcp_tools = await client.get_tools()

    ### Defining llm model
    # llm_model_1 = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    # llm_model_2 = ChatOllama(model="llama3.2")

    llm_model_1 = ChatOllama(model="llama3.2")
    # llm_model_1 = ChatGroq(model="llama-3.3-70b-versatile")

    ### Stock Finder Agent
    stock_finder_agent = create_agent(
        model=llm_model_1, 
        tools=mcp_tools,
        system_prompt=system_prompts["STOCK_FINDER_AGENT"],
        name="stock_finder_agent"
    )

    ### Market Data Agent
    market_data_agent = create_agent(
        model=llm_model_1, 
        tools=mcp_tools,
        system_prompt=system_prompts["MARKET_DATA_AGENT"],
        name="market_data_agent"
    )

    ### News Analyst Agent
    news_analysis_agent = create_agent(
        model=llm_model_1, 
        tools=mcp_tools,
        system_prompt=system_prompts["NEWS_ANALYSIS_AGENT"],
        name="news_analysis_agent"
    )

    ### Price Recommender Agent
    price_recommender_agent = create_agent(
        model=llm_model_1, 
        tools=mcp_tools,
        system_prompt=system_prompts["PRICE_RECOMMENDER_AGENT"],
        name="price_recommender_agent"
    )

    ### Supervisor Agent
    supervisor_agent = create_supervisor(
        model=llm_model_1,
        agents=[stock_finder_agent, market_data_agent, news_analysis_agent, price_recommender_agent],
        system_prompt=system_prompts["SUPERVISOR_AGENT"],
        output_mode="full_history",
        add_handoff_back_messages=True
    ).compile()

    # Use astream() for async iteration with MCP tools
    async for chunk in supervisor_agent.astream(
        {"messages": [{"role": "user", "content": user_input}]}
    ):
        pretty_print_messages(chunk, last_message=True)

if __name__ == "__main__":
    asyncio.run(run_agent("Distribute the task to the four special agents and using that information, give me good stock recommendation from NSE analysis the market trends, stocks and news sentiment. Dont answer directly, use the agents intelligently "))