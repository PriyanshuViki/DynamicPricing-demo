import os
import streamlit as st
from strands import agent
from strands.models import BedrockModel

from tools.pricing_tools import (
    get_revenue_summary,
    get_material_info,
    get_top_opportunities,
    get_pricing_status,
    get_margin_analysis,
    get_material_groups,
    search_materials,
    get_underpriced_materials,
    get_price_ranges,
    get_quantity_analysis
)

# Load credentials from Streamlit secrets (cloud) or .env (local)
try:
    # Try Streamlit secrets first (for cloud deployment)
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = st.secrets["AWS_DEFAULT_REGION"]
    MODEL_ID = st.secrets["MODEL_ID"]
except:
    # Fall back to .env file (for local development)
    from dotenv import load_dotenv
    load_dotenv()
    MODEL_ID = os.getenv("MODEL_ID")


# Configure Bedrock Model (uses environment variables automatically)
bedrock = BedrockModel(
    model_id=MODEL_ID,
    temperature=0.3,
)

SYSTEM_PROMPT = """
You are a pricing analytics assistant helping analyze SAP sales data and pricing strategies.

You have access to tools to query:
- Revenue summaries and opportunities
- Material information and pricing
- Pricing status (underpriced/optimal/overpriced)
- Margin analysis
- Material groups
- Search capabilities

Guidelines:
- Use appropriate tools to fetch data before answering
- Format currency as ₹X.XX Crores or ₹X.XXM
- Provide specific numbers and actionable insights
- Be concise and business-focused
- If user asks about visualizations, mention that charts can be generated
- Always use tools to get accurate data rather than guessing

When users ask about materials, pricing, margins, or revenue, use the relevant tools to fetch accurate data.
"""

pricing_agent = agent.Agent(
    name="pricing_analytics_agent",
    model=bedrock,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        get_revenue_summary,
        get_material_info,
        get_top_opportunities,
        get_pricing_status,
        get_margin_analysis,
        get_material_groups,
        search_materials,
        get_underpriced_materials,
        get_price_ranges,
        get_quantity_analysis
    ],
)

def process_query(user_query: str):
    """Process user query through the agent"""
    response = pricing_agent(user_query)
    return response
