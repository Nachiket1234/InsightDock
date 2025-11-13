# agent_core.py
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import duckdb
import google.generativeai as genai
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """State container for the agent workflow"""
    messages: List[Dict[str, Any]]
    current_tool: str = ""
    tool_input: str = ""
    tool_output: str = ""
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class OlistGenAIAgent:
    def __init__(self, db_path: str = "olist_fixed.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.setup_gemini()
        self.setup_tools()
        self.setup_rag()
        self.memory = ConversationBufferWindowMemory(k=10)
        
    def setup_gemini(self):
        """Initialize Gemini model with function calling"""
        # In production, this would come from environment variables
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCkGMwyO0t5P8JTGQTxQngCBr9HtRbv7-I")
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=self.generation_config,
        )
        
    def setup_tools(self):
        """Initialize all tools for the agent"""
        self.tools = {
            "sql_analyst": self.create_sql_tool(),
            "python_repl": self.create_python_tool(),
            "web_search": self.create_web_search_tool(),
            "image_generator": self.create_image_tool(),
            "map_generator": self.create_map_tool(),
            "translator": self.create_translator_tool(),
            "sentiment_analyzer": self.create_sentiment_tool(),
        }
        
    def create_sql_tool(self) -> Tool:
        """Tool for executing SQL queries on Olist database"""
        def run_sql_query(query: str) -> str:
            try:
                # Security: Basic validation to prevent destructive operations
                destructive_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
                if any(keyword in query.upper() for keyword in destructive_keywords):
                    return "Error: Destructive operations are not allowed."
                
                result = self.conn.execute(query).df()
                
                if result.empty:
                    return "No results found."
                
                # Return summary for large results
                if len(result) > 100:
                    summary = f"Found {len(result)} rows. Showing first 10:\n"
                    summary += result.head(10).to_string()
                    return summary
                
                return result.to_string()
                
            except Exception as e:
                return f"SQL Error: {str(e)}"
        
        return Tool(
            name="SQL Analyst",
            description="Execute SQL queries on the Brazilian E-commerce dataset. Use for sales analysis, customer insights, product trends.",
            func=run_sql_query
        )
    
    def create_python_repl_tool(self) -> Tool:
        """Tool for executing Python code for complex analysis"""
        def run_python_code(code: str) -> str:
            try:
                # Security: Restricted execution environment
                allowed_modules = ['pandas', 'numpy', 'matplotlib', 'plotly', 'datetime']
                restricted_builtins = ['__import__', 'eval', 'exec', 'open', 'file']
                
                # Create safe environment
                safe_globals = {
                    'pd': __import__('pandas'),
                    'np': __import__('numpy'),
                    'plt': __import__('matplotlib.pyplot'),
                    'px': __import__('plotly.express'),
                    'go': __import__('plotly.graph_objects'),
                    'datetime': __import__('datetime'),
                }
                
                safe_locals = {}
                
                # Execute code
                exec(code, safe_globals, safe_locals)
                
                # Return any result variable
                if 'result' in safe_locals:
                    return str(safe_locals['result'])
                else:
                    return "Code executed successfully."
                    
            except Exception as e:
                return f"Python Error: {str(e)}"
        
        return Tool(
            name="Python REPL",
            description="Execute Python code for complex data analysis, visualizations, and statistical computations.",
            func=run_python_code
        )
    
    def create_web_search_tool(self) -> Tool:
        """Tool for web search using Tavily"""
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY", "your_tavily_key")
            search = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=3)
            return search
        except:
            # Fallback to simple web search
            return Tool(
                name="Web Search",
                description="Search the web for current information about products, market trends, or business context.",
                func=lambda x: "Web search unavailable. Please check API key."
            )
    
    def create_image_tool(self) -> Tool:
        """Tool for generating images using Gemini"""
        def generate_image(prompt: str) -> str:
            try:
                # In a real implementation, this would call Gemini's image generation
                return f"Image generated for: {prompt}. [Image generation would appear here in production]"
            except Exception as e:
                return f"Image generation error: {e}"
        
        return Tool(
            name="Image Generator", 
            description="Generate product images, marketing visuals, or data visualizations.",
            func=generate_image
        )
    
    def create_map_tool(self) -> Tool:
        """Tool for generating geographic maps"""
        def generate_map(locations_data: str) -> str:
            try:
                # Parse location data and create map
                # This is a simplified version - real implementation would parse coordinates
                m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)  # Brazil center
                folium.Marker([-23.5505, -46.6333], popup='São Paulo').add_to(m)
                return "Map generated. [Interactive map would appear in Streamlit]"
            except Exception as e:
                return f"Map generation error: {e}"
        
        return Tool(
            name="Map Generator",
            description="Create geographic maps showing customer locations, sales distribution, or delivery routes.",
            func=generate_map
        )
    
    def create_translator_tool(self) -> Tool:
        """Tool for translating Portuguese reviews"""
        def translate_text(text: str) -> str:
            try:
                # Simple translation using Gemini
                prompt = f"Translate this Portuguese text to English: {text}"
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Translation error: {e}"
        
        return Tool(
            name="Translator",
            description="Translate Portuguese text to English, especially useful for product reviews.",
            func=translate_text
        )
    
    def create_sentiment_tool(self) -> Tool:
        """Tool for sentiment analysis of reviews"""
        def analyze_sentiment(text: str) -> str:
            try:
                prompt = f"Analyze the sentiment of this review (positive/negative/neutral) and provide a score from 1-5: {text}"
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Sentiment analysis error: {e}"
        
        return Tool(
            name="Sentiment Analyzer",
            description="Analyze sentiment of product reviews to understand customer satisfaction.",
            func=analyze_sentiment
        )
    
    def setup_rag(self):
        """Setup RAG system for Portuguese reviews"""
        try:
            # Load and chunk review data
            reviews_df = self.conn.execute("""
                SELECT review_id, review_comment_message 
                FROM order_reviews 
                WHERE review_comment_message IS NOT NULL 
                AND LENGTH(review_comment_message) > 10
                LIMIT 10000  # Limit for demo
            """).df()
            
            # Create documents for RAG
            documents = []
            for _, row in reviews_df.iterrows():
                if pd.notna(row['review_comment_message']):
                    doc = Document(
                        page_content=row['review_comment_message'],
                        metadata={"review_id": row['review_id']}
                    )
                    documents.append(doc)
            
            # Initialize embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="./chroma_reviews"
            )
            
            logger.info(f"RAG system initialized with {len(documents)} reviews")
            
        except Exception as e:
            logger.error(f"RAG setup failed: {e}")
            self.vector_store = None
    
    def rag_search(self, query: str, k: int = 3) -> List[Document]:
        """Search reviews using RAG"""
        if not self.vector_store:
            return []
        
        try:
            # Translate query to Portuguese for better matching
            translated_query = self.tools["translator"].func(f"Translate to Portuguese: {query}")
            
            docs = self.vector_store.similarity_search(translated_query, k=k)
            return docs
        except:
            # Fallback to English search
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
    
    def process_user_query(self, user_input: str, state: AgentState) -> AgentState:
        """Main method to process user queries through the agent system"""
        try:
            # Update state with user message
            state.messages.append({"role": "user", "content": user_input})
            
            # Determine which tool to use based on query
            tool_selection_prompt = f"""
            User Query: {user_input}
            Available Tools: {list(self.tools.keys())}
            
            Based on the user's query, select the most appropriate tool and provide the exact input for that tool.
            Format: TOOL_NAME|TOOL_INPUT
            
            Examples:
            - "Show me sales by category" -> "sql_analyst|SELECT product_category_name_english, SUM(total_amount) as revenue FROM master_olist_clean GROUP BY product_category_name_english ORDER BY revenue DESC LIMIT 10"
            - "Translate this Portuguese review: Muito bom!" -> "translator|Muito bom!"
            - "Search for current electronics market trends" -> "web_search|electronics market trends Brazil 2024"
            
            Response:
            """
            
            response = self.model.generate_content(tool_selection_prompt)
            tool_selection = response.text.strip()
            
            # Parse tool selection
            if "|" in tool_selection:
                state.current_tool, state.tool_input = tool_selection.split("|", 1)
                state.current_tool = state.current_tool.strip()
                state.tool_input = state.tool_input.strip()
            else:
                # Default to SQL analyst for data queries
                state.current_tool = "sql_analyst"
                state.tool_input = user_input
            
            # Execute tool
            if state.current_tool in self.tools:
                state.tool_output = self.tools[state.current_tool].func(state.tool_input)
            else:
                state.tool_output = f"Tool '{state.current_tool}' not found."
            
            # Generate final response
            final_prompt = f"""
            User Query: {user_input}
            Tool Used: {state.current_tool}
            Tool Input: {state.tool_input}
            Tool Output: {state.tool_output}
            
            Based on the tool output, provide a helpful, conversational response to the user.
            Explain the results in simple terms and offer additional insights if relevant.
            
            Response:
            """
            
            final_response = self.model.generate_content(final_prompt)
            
            # Update state with assistant response
            state.messages.append({
                "role": "assistant", 
                "content": final_response.text,
                "tool_used": state.current_tool,
                "tool_output": state.tool_output
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            state.messages.append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}. Please try rephrasing your question."
            })
            return state

# Example usage
if __name__ == "__main__":
    # Test the agent
    agent = OlistGenAIAgent("olist_fixed.duckdb")
    
    # Test queries
    test_queries = [
        "What are the top 5 product categories by revenue?",
        "Show me sales trends over time",
        "Translate this review: 'Produto excelente, entrega rápida'",
        "What's the average customer review score?"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Query: {query}")
        state = AgentState(messages=[])
        result_state = agent.process_user_query(query, state)
        print(f"🤖 Response: {result_state.messages[-1]['content']}")