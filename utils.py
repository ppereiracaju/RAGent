import time
from anthropic import Anthropic
import logging
import requests
from tavily import TavilyClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the Anthropic client
client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))

def call_claude_rag(query, context):
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
    If the context doesn't contain enough information to answer confidently, indicate that.
    
    Context:
    {context}
    """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=system_prompt,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return message.content

def call_tavily_web_search(query):
    # Instantiating TavilyClient with API key from environment
    tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

    # Step 2. Executing a context search query
    context = tavily_client.get_search_context(query=query)

    # Step 3. That's it! You now have a context string that you can feed directly into your RAG Application
    return context

def assess_confidence(response):
    """
    Use Claude to evaluate the confidence of the response.
    Returns True if confident, False if not confident.
    """
    system_prompt = """Evaluate the confidence level of the following response.
    Consider:
    1. Completeness of the answer
    2. Specificity and precision
    3. Presence of uncertainty markers (e.g., "might", "maybe", "I'm not sure")
    4. Consistency of information
    
    Return a JSON with two fields:
    - confidence_score: (float between 0 and 1)
    - reasoning: (brief explanation)
    """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Response to evaluate: {response}"}
        ]
    )
    
    try:
        evaluation = message.content
        # Extract confidence score from the response
        if '"confidence_score":' in evaluation:
            confidence_score = float(evaluation.split('"confidence_score":')[1].split(',')[0].strip())
            return confidence_score > 0.7  # Return True if confidence is above 70%
        return False
    except Exception as e:
        logging.error(f"Error in confidence assessment: {e}")
        return False

def synthesize_information(internal_response, web_results):
    system_prompt = """Synthesize the internal knowledge and web search results into a comprehensive response. 
    Clearly indicate which information comes from which source."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Internal Knowledge: {internal_response}\n\nWeb Results: {web_results}"}
        ]
    )
    return message.content 