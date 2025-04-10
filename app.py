import streamlit as st
import asyncio
from autogen import AssistantAgent, UserProxyAgent
from agenticai import PredictionFinder, PredictorProfiler, PredictionVerifier
import os 
from groq import Groq

# API Keys - Replace with your actual keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
DATURA_API_KEY = os.environ.get("DATURA_API_KEY", "dt_$X6oACKtNOE_2RL984Dg-C8Ds6HZmsQLA4N7ez3NysVg")
NEWS_API_TOKEN = os.environ.get("NEWS_API_TOKEN", "drAk0dGvkyZWSoutZe1sRgfY81HpTYiwERgrSgsw")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBo8-CKyb3IzZbRzx685TqDi9EutAg7FkE")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "64c807de4a9d1425d")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

OPEN_AI_KEY = "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"
# Constants openai/gpt-4.5-preview openai/gpt-3.5-turbo
MODEL_NAME = "llama-3.3-70b-versatile"  
MODEL_NAME_1 = "openai/gpt-4.5-preview"
DATURA_API_URL = "https://apis.datura.ai/twitter"


# ============ AUTOGEN INTEGRATION ============
# Register the functions with the agents

function_definitions = [
    {
        "name": "find_predictions",
        "description": "Finds predictions based on a user prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "The prompt provided by the user to find predictions."
                }
            },
            "required": ["user_prompt"]
        }
    },
    {
        "name": "build_profile",
        "description": "Builds a profile for a given handle.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "The handle (e.g., username or identifier) for whom to build a profile."
                }
            },
            "required": ["handle"]
        }
    },
    {
        "name": "calculate_credibility",
        "description": "Calculates the credibility score for a given handle.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "The handle (e.g., username or identifier) for whom to calculate the credibility score."
                }
            },
            "required": ["handle"]
        }
    },
    {
        "name": "verify_prediction",
        "description": "Verifies a given prediction.",
        "parameters": {
            "type": "object",
            "properties": {
                "prediction": {
                    "type": "string",
                    "description": "The prediction text to be verified."
                }
            },
            "required": ["prediction"]
        }
    }
]


tools_schema = [
    {"type": "function", "function": func_def} for func_def in function_definitions
]


st.write("""# AutoGen Chat Agents""")

class StreamlitAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

class StreamlitUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


# Define Autogen agents
def create_prediction_agents():
    # Configuration for the LLM seed = 42 (Store the previous interactions)
    llm_config = {
        "config_list": [{"model": MODEL_NAME_1, "api_key": OPEN_AI_KEY, "base_url": "https://openrouter.ai/api/v1"}],
        "temperature": 0.3,
        "tools": tools_schema,
    }
    
    # Initialize components
    prediction_finder = PredictionFinder(groq_client, DATURA_API_KEY, DATURA_API_URL)
    predictor_profiler = PredictorProfiler(groq_client, DATURA_API_KEY, DATURA_API_URL)
    prediction_verifier = PredictionVerifier(groq_client, NEWS_API_TOKEN, GOOGLE_API_KEY, GOOGLE_CSE_ID)

    def find_predictions_wrapper(user_prompt: str):
        """Wrapper for the find_predictions function"""
        print("Finding predictions...")
        return prediction_finder.find_predictions(user_prompt)

    def build_profile_wrapper(handle: str):
        """Wrapper for the build_profile function"""
        print("Building profile...")
        return predictor_profiler.build_profile(handle)

    def calculate_credibility_wrapper(handle: str):
        print("Calculating credibility score...")
        """Wrapper for the calculate_credibility_score function"""
        return predictor_profiler.calculate_credibility_score(handle, prediction_verifier)

    def verify_prediction_wrapper(prediction: str):
        print("Verifying prediction...")    
        """Wrapper for the verify_prediction function"""
        return prediction_verifier.verify_prediction(prediction)

    # Create function map for the UserProxyAgent with the new function
    function_map = {
        "find_predictions": find_predictions_wrapper,
        "build_profile": build_profile_wrapper,
        "verify_prediction": verify_prediction_wrapper,
        "calculate_credibility": calculate_credibility_wrapper
    }
    
#     # Create the assistant agent with updated system message
#     assistant = AssistantAgent(
#         name="PredictionAssistant",
#         llm_config=llm_config,
#         system_message="""You are a prediction analysis expert that helps users find, profile, and verify predictions.
# You work with these four main functions:
# 1. find_predictions(user_prompt) - Finds X posts containing predictions on a topic
# 2. build_profile(handle) - Builds a profile of a predictor based on their history
# 3. verify_prediction(prediction_query) - Verifies if a prediction came true
# 4. calculate_credibility(handle) - Calculates a credibility score for a predictor based on verified predictions

# Help the user analyze predictions according to their needs. Always think step-by-step and suggest the best approach.
#         """
#     )
    
#     # Create the user proxy agent
#     user_proxy = UserProxyAgent(
#         name="PredictionAnalyst",
#         human_input_mode="ALWAYS",
#         function_map=function_map,
#         code_execution_config={
#         "work_dir": "prediction-swarm",
#         "use_docker": False,  # Set to True if you want to run in Docker
#         },
#     )
    # Create the assistant agent with strict instructions
    assistant = AssistantAgent(
        name="PredictionAssistant",
        llm_config=llm_config,
        system_message="""You are a prediction analysis expert that helps users find, profile, and verify predictions.
        
STRICT RULES YOU MUST FOLLOW:
1. You MUST ONLY use the provided functions - never make up data or predictions
2. You MUST ask clarifying questions if the user request is unclear
3. You MUST verify all predictions before making claims about accuracy
4. You MUST direct the user to choose one of the four function options

AVAILABLE FUNCTIONS:
1. find_predictions(user_prompt) - Finds posts containing predictions on a topic 
2. build_profile(handle) - Builds a profile of a predictor based on their history
3. verify_prediction(prediction_query) - Verifies if a prediction came true
4. calculate_credibility(handle) - Calculates a credibility score for a predictor

Respond ONLY with one of these approaches:
- Ask clarifying questions if needed
- Propose which function to use based on the user's need
- Execute one of the functions with appropriate parameters
- Present results from function calls (never make up data)
"""
    )
    
    # Create the user proxy agent with execution disabled
    user_proxy = UserProxyAgent(
        name="PredictionAnalyst",
        human_input_mode="TERMINATE",
        #max_consecutive_auto_reply=2,
        function_map=function_map,
        code_execution_config={
        "suppress_stdout": True,
        "auto_reply": True,  # Add this line to automatically execute function calls    
        "work_dir": "prediction-swarm",
        "use_docker": False,  # Set to True if you want to run in Docker
        },
    )
    
    return assistant, user_proxy    
    


selected_model = None
selected_key = None
with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
    selected_key = st.text_input("API Key", type="password")

with st.container():
    # for message in st.session_state["messages"]:
    #    st.markdown(message)

    user_input = st.chat_input("Ask about predictions...")
    if user_input:
        assistant, user_proxy = create_prediction_agents()

        # Async chat handling
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def initiate_chat():
            await user_proxy.a_initiate_chat(assistant, message=user_input)

        loop.run_until_complete(initiate_chat())
