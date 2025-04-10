import os
import json
import requests
import re
import time
from typing import List, Dict, Any, Tuple
from groq import Groq   
from autogen import Agent, AssistantAgent, UserProxyAgent, ConversableAgent
import os
os.environ["AUTOGEN_DEBUG"] = "1"  # Basic debug info
os.environ["AUTOGEN_VERBOSE"] = "1"  # More detailed logging1
import streamlit as st

"""
# API Keys - Replace with your actual keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
DATURA_API_KEY = os.environ.get("DATURA_API_KEY", "dt_$X6oACKtNOE_2RL984Dg-C8Ds6HZmsQLA4N7ez3NysVg")
NEWS_API_TOKEN = os.environ.get("NEWS_API_TOKEN", "drAk0dGvkyZWSoutZe1sRgfY81HpTYiwERgrSgsw")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBo8-CKyb3IzZbRzx685TqDi9EutAg7FkE")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "64c807de4a9d1425d")
OPEN_AI_KEY = os.environ.get(OPEN_AI_KEY, "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")
"""

# Load keys from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DATURA_API_KEY = st.secrets["DATURA_API_KEY"]
NEWS_API_TOKEN = st.secrets["NEWS_API_TOKEN"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
OPEN_AI_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Constants openai/gpt-4.5-preview openai/gpt-3.5-turbo
MODEL_NAME = "llama-3.3-70b-versatile"  
MODEL_NAME_1 = "openai/gpt-4.5-preview"
DATURA_API_URL = "https://apis.datura.ai/twitter"


# ============ COMPONENT 1: PREDICTION FINDER ============

class PredictionFinder:
    """Finds tweets containing predictions about specified topics."""
    
    def __init__(self, groq_client, datura_api_key, datura_api_url):
        self.groq_client = groq_client
        self.datura_api_key = datura_api_key
        self.datura_api_url = datura_api_url
    
    def generate_search_query(self, user_prompt: str) -> str:
        """Generate a properly formatted search query from user prompt."""
        context = """You are an expert in constructing search queries for the Datura API to find relevant tweets related to Polymarket predictions.
Your task is to generate properly formatted queries based on user prompts.

Here are some examples of well-structured Datura API queries:

1. I want mentions of predictions about the 2024 US Presidential Election, excluding Retweets, with at least 20 likes.  
   Query: (President) (elections) (USA) min_faves:20  

2. I want tweets from @Polymarket users discussing cryptocurrency price predictions, excluding tweets without links.  
   Query: (Bitcoin) (Ethereum) (crypto) 

3. I want tweets predicting the outcome of the Wisconsin Supreme Court election between Susan Crawford and Brad Schimel.  
   Query: (Wisconsin) (SupremeCourt) (Crawford) (Schimel) (election)

4. I want tweets discussing AI stock price predictions in 2025  
   Query: (AI) (tech) (stock)

5. I want mentions of predictions about the winner of the 2025 NCAA Tournament.  
   Query: (NCAA) (MarchMadness) (2025) (winner) 

6. I want tweets discussing whether Yoon will be out as president of South Korea before May.  
   Query: (Yoon) (SouthKorea) (president) (resign) (before May) 

Now, given the following user prompt, generate a properly formatted Datura API query. (Just the query, no additional text or explanation.)"""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return completion.choices[0].message.content.strip()
    
    def get_tweets(self, query: str, min_likes: int = 100, count: int = 50) -> List[Dict]:
        """Fetch tweets from Datura API based on the generated query."""
        payload = {
            "query": query,
            "sort": "Top",
            "start_date": "2024-02-25",
            "lang": "en",
            "verified": True,
            "blue_verified": True,
            "is_quote": False,
            "is_video": False,
            "is_image": False,
            "min_retweets": 100,
            "min_replies": 10,
            "min_likes": min_likes,
            "count": count
        }
        
        headers = {
            "Authorization": self.datura_api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.get(self.datura_api_url, params=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching tweets: {response.status_code}")
            return []
    
    def process_tweets(self, tweets: List[Dict]) -> Tuple[Dict, Dict]:
        """Process tweets to create structured data."""
        hash_dict = {}
        username_to_tweet = {}
        
        for tweet in tweets:
            username = tweet["user"]["id"]
            
            hash_dict[username] = {
                "username": tweet["user"]["username"],
                "favourites_count": tweet["user"]["favourites_count"],
                "is_blue_verified": tweet["user"]["is_blue_verified"],
                "tweet_text": tweet["text"],
                "like_count": tweet["like_count"],
                "created_at": tweet["created_at"],
            }
            
            username_to_tweet[tweet["user"]["username"]] = tweet["text"]
        
        return hash_dict, username_to_tweet
    
    def analyze_predictions(self, username_to_tweet: Dict) -> str:
        """Analyze tweets to identify predictions."""
        json_string = json.dumps(username_to_tweet, indent=4)
        
        context = """You are an expert in identifying explicit and implicit predictions in tweets related to Polymarket topics.

Here is a JSON object containing tweets, where each key represents a unique tweet ID and the value is the tweet text.

Your task:  
For each tweet, determine if it contains an **explicit or implicit prediction** about a future event **related to a Polymarket topic**.  
- If it **does**, return "Yes".  
- If it **does not**, return "No".  

Format your response as a JSON object with each tweet ID mapped to "Yes" or "No".

Example:

Input JSON:
{
    "username1": "Bitcoin will hit $100K by the end of 2025!",
    "username2": "The economy is in trouble. People are struggling.",
    "username3": "I bet Trump wins the next election."
}

Expected Output:
{
    "username1": "Yes",
    "username2": "No",
    "username3": "Yes"
}

Now, analyze the following tweets and generate the output: 
Ensure the response is **valid JSON** with no additional text.
"""
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": json_string}
            ]
        )
        
        return completion.choices[0].message.content
    
    def filter_tweets_by_prediction(self, yes_no: str, hash_dict: Dict) -> str:
        """Filter tweets to only include those with predictions."""
        match_yes_no = re.search(r"\{(.*)\}", yes_no, re.DOTALL)
        json_content_yes_no = "{" + match_yes_no.group(1) + "}"
        
        yes_no_dict = json.loads(json_content_yes_no)
        
        filtered_tweets = {
            tweet_id: details
            for tweet_id, details in hash_dict.items()
            if details["username"] in yes_no_dict and yes_no_dict[details["username"]] == "Yes"
        }
        
        return json.dumps(filtered_tweets, indent=4)
    
    def find_predictions(self, user_prompt: str) -> Dict:
        """Main method to find predictions based on user prompt."""
        # Generate search query
        query = self.generate_search_query(user_prompt)
        print(f"Generated Search Query: {query}")
        
        # Get tweets
        tweets = self.get_tweets(query)
        
        if not tweets:
            return {"error": "No tweets found matching the criteria"}
        
        # Process tweets
        hash_dict, username_to_tweet = self.process_tweets(tweets)
        
        # Analyze predictions
        prediction_analysis = self.analyze_predictions(username_to_tweet)
        
        # Filter tweets
        filtered_predictions = self.filter_tweets_by_prediction(prediction_analysis, hash_dict)
        print("Filtered predictions:", len(filtered_predictions))
        # Return as dictionary
        return json.loads(filtered_predictions)


# ============ COMPONENT 2: PREDICTOR PROFILE BUILDER ============

class PredictorProfiler:
    """Builds comprehensive profiles of predictors based on their prediction history."""
    
    def __init__(self, groq_client, datura_api_key, datura_api_url):
        self.groq_client = groq_client
        self.datura_api_key = datura_api_key
        self.datura_api_url = datura_api_url
    
    def build_user_profile(self, handle: str, max_retries: int = 5) -> Dict:
        """Fetch recent tweets from a specific user."""
        headers = {
            "Authorization": f"{self.datura_api_key}",
            "Content-Type": "application/json",
        }
        
        params = {
            "query": f"from:{handle}",
            "sort": "Top",
            "lang": "en",
            "verified": True,
            "blue_verified": True,
            "is_quote": False,
            "is_video": False,
            "is_image": False,
            "min_retweets": 100,
            "min_replies": 10,
            "min_likes": 100,
            "count": 30 #100
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.datura_api_url, params=params, headers=headers)
                response.raise_for_status()
                tweets_ls = response.json()
                print(len(tweets_ls), "tweets found") # Amit
                if tweets_ls:
                    tweets = [tweet.get("text", "") for tweet in tweets_ls]
                    raw_tweets = tweets_ls
                    return {"tweets": tweets, "raw_tweets": raw_tweets}
                
            except requests.exceptions.RequestException as e:
                return {"error": f"Failed to fetch tweets: {str(e)}", "tweets": [], "raw_tweets": []}
            
            print(f"Attempt {attempt + 1} failed. Retrying...")
            time.sleep(2)
        
        return {"error": "Invalid Username. No tweets found after 5 attempts.", "tweets": [], "raw_tweets": []}
    
    def filter_predictions(self, tweets: List[str]) -> Dict:
        """Filter tweets to only include predictions."""
        tweets = tweets[:30]  # Limit to 30 tweets for analysis

        tweet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tweets)])
        
        system_context = """You are an expert in identifying explicit and implicit predictions in tweets that could be relevant to Polymarket, a prediction market platform. Polymarket users bet on future events in politics, policy, business, law, and geopolitics.

    **Definitions:**
    1. **Explicit Prediction**: A direct statement about a future outcome (e.g., 'X will happen,' 'Y is likely to pass').
    2. **Implicit Prediction**: A statement implying a future outcome (e.g., 'Senator proposes bill,' 'Protests may lead to...').

    **Polymarket Topics Include:**
    - Elections, legislation, court rulings
    - Policy changes (tariffs, regulations)
    - Business decisions (company moves, market impacts)
    - Geopolitical events (wars, treaties, sanctions)
    - Legal/Investigative outcomes (prosecutions, declassifications)

    **Exclude:**
    - Past events (unless they imply future consequences)
    - Pure opinions without forecastable outcomes
    - Non-actionable statements (e.g., 'People are struggling')

    **Examples:**
    - 'Trump will win in 2024' → **Yes (Explicit)**
    - 'Senator proposes bill to ban TikTok' → **Yes (Implicit)**
    - 'The economy is collapsing' → **No (No actionable prediction)**

    **Task:** For each tweet, return **'Yes'** if it contains an explicit/implicit prediction relevant to Polymarket, else **'No'**. Respond *only* with a JSON object like:
    {
    "predictions": ["Yes", "No", ...]
    }
    """
        
        response = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": tweet_list}
            ]
        )
        
        raw_output = response.choices[0].message.content
        
        # Remove markdown wrapping if present
        if raw_output.startswith("```json"):
            raw_output = re.sub(r"```json|```", "", raw_output).strip()
        
        try:
            parsed = json.loads(raw_output)
            return {
                "predictions": parsed.get("predictions", []),
            }
        except Exception as e:
            print("Failed to parse LLM response:")
            print(raw_output)
            raise e
    
    def apply_filter(self, tweets: List[str], outcomes: Dict) -> List[str]:
        """Apply prediction filter to tweets."""
        outcomes_list = outcomes["predictions"]
        zipped = list(zip(tweets, outcomes_list))
        filtered_tweets = [tweet for tweet, outcome in zipped if outcome == "Yes"]
        print(f"Filtered {len(filtered_tweets)} prediction tweets from {len(tweets)} total tweets.")
        return filtered_tweets
    
    def analyze_prediction_patterns(self, filtered_tweets: List[str]) -> Dict:
        """Analyze patterns in the user's predictions."""
        if not filtered_tweets:
            return {
                "total_predictions": 0,
                "topics": {},
                "confidence_level": "N/A",
                "prediction_style": "N/A",
                "summary": "No predictions found for this user."
            }
        
        tweet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(filtered_tweets)])
        
        analysis_prompt = f"""
        You are an expert analyst of prediction patterns and behaviors.  
        Analyze the following list of prediction tweets from a single user and provide a comprehensive analysis with the following information:

        1. The main topics this person makes predictions about (politics, crypto, sports, etc.)
        2. Their typical confidence level (certain, hedging, speculative)
        3. Their prediction style (quantitative, qualitative, conditional)
        4. Any patterns you notice in their prediction behavior

        Format your response as JSON:
        {{
            "topics": {{"topic1": percentage, "topic2": percentage, ...}},
            "confidence_level": "description of their confidence level",
            "prediction_style": "description of their prediction style",
            "patterns": ["pattern1", "pattern2", ...],
            "summary": "A brief summary of this predictor's profile"
        }}

        Ensure the response is **valid JSON** with no additional text.
        """
        
        response = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": tweet_list}
            ]
        )
        
        raw_output = response.choices[0].message.content
        
        # Extract JSON from response
        match = re.search(r"\{(.*)\}", raw_output, re.DOTALL)
        json_content = "{" + match.group(1) + "}"
        
        try:
            analysis = json.loads(json_content)
            analysis["total_predictions"] = len(filtered_tweets)
            return analysis
        except json.JSONDecodeError:
            return {
                "total_predictions": len(filtered_tweets),
                "error": "Could not parse analysis",
                "raw_output": raw_output
            }
    
    def build_profile(self, handle: str) -> Dict:
        """Main method to build a predictor's profile."""
        # Get user tweets
        user_data = self.build_user_profile(handle)
        
        if "error" in user_data:
            return {"error": user_data["error"]}
        
        # Filter predictions
        prediction_outcomes = self.filter_predictions(user_data["tweets"])
        
        # Apply filter
        filtered_predictions = self.apply_filter(user_data["tweets"], prediction_outcomes)
        print("Filtered predictions build profile:", len(filtered_predictions))
        # Analyze prediction patterns
        analysis = self.analyze_prediction_patterns(filtered_predictions)
        
        # Build complete profile
        profile = {
            "handle": handle,
            "total_tweets_analyzed": len(user_data["tweets"]),
            "prediction_tweets": filtered_predictions,
            "prediction_count": len(filtered_predictions),
            "prediction_rate": len(filtered_predictions) / len(user_data["tweets"]) if user_data["tweets"] else 0,
            "analysis": analysis
        }
        
        return profile

    def calculate_credibility_score(self, handle: str, prediction_verifier: object) -> Dict:
        """Calculate credibility score based on verification of user's predictions."""
        # First get the user profile with predictions
        profile = self.build_profile(handle)
        
        if "error" in profile:
            return {"error": profile["error"]}
        
        if not profile["prediction_tweets"]:
            return {
                "handle": handle,
                "credibility_score": 0.0,
                "prediction_stats": {
                    "total": 0,
                    "true": 0,
                    "false": 0,
                    "uncertain": 0
                },
                "message": "No predictions found for this user."
            }
        
        # Track verification results
        verification_stats = {
            "total": len(profile["prediction_tweets"]),
            "true": 0,
            "false": 0,
            "uncertain": 0,
            "verifications": []
        }
        
        # Verify each prediction
        for prediction in profile["prediction_tweets"]:
            # Verify the prediction
            verification = prediction_verifier.verify_prediction(prediction)
            
            # Update stats
            if verification["result"] == "TRUE":
                verification_stats["true"] += 1
            elif verification["result"] == "FALSE":
                verification_stats["false"] += 1
            else:  # UNCERTAIN
                verification_stats["uncertain"] += 1
            
            # Store verification result with the prediction
            verification_stats["verifications"].append({
                "prediction": prediction,
                "result": verification["result"],
                "summary": verification["summary"],
                "sources": verification["sources"]
            })
        
        # Calculate credibility score
        if verification_stats["true"] + verification_stats["false"] + verification_stats["uncertain"] > 0:
            credibility_score = verification_stats["true"] / verification_stats["total"]
        else:
            credibility_score = 0.0
        
        # Create the final result
        result = {
            "handle": handle,
            "credibility_score": round(credibility_score, 2),
            "prediction_stats": {
                "total": verification_stats["total"],
                "true": verification_stats["true"],
                "false": verification_stats["false"],
                "uncertain": verification_stats["uncertain"]
            },
            "verified_predictions": verification_stats["verifications"],
            "profile_summary": profile["analysis"]["summary"] if "summary" in profile["analysis"] else ""
        }
        
        return result

# ============ COMPONENT 3: PREDICTION VERIFICATION ============

class PredictionVerifier:
    """Verifies whether predictions have come true or proven false."""
    
    def __init__(self, groq_client, news_api_token, google_api_key, google_cse_id):
        self.groq_client = groq_client
        self.news_api_token = news_api_token
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
    
    def fetch_google_results(self, query: str) -> List[Dict]:
        """Fetch search results from Google Custom Search API."""
        google_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.google_api_key}&cx={self.google_cse_id}&num=3"
        
        response = requests.get(google_url)
        if response.status_code == 200:
            data = response.json()
            return data.get("items", [])
        return []
    
    def generate_search_query(self, prediction_query: str) -> str:
        """Generate a search query for news APIs based on the prediction."""
        context = """
        You are an expert in constructing search queries for news APIs to find relevant articles related to political predictions.
        Your task is to generate a properly formatted query for searching news related to a given prediction.

        Examples:
        1. Prediction: 'Chances of UK leaving the European Union in 2016 was 52%'
           Query: Brexit, UK, European Union, 2016

        2. Prediction: 'Chances of Donald Trump winning the 2016 US Presidential Election was 30%'
           Query: Donald Trump, elections, 2024

        3. Prediction: 'Chances of Apple's iPhone revolutionizing the smartphone industry in 2007 was 80%'
           Query: Apple, iPhone, smartphone, 2007

        4. Prediction: 'Chances of India winning T20 Cricket WorldCup Final in 2024 was 52%'
           Query: India, T20, Cricket World Cup, Winner, 2024

        5. Prediction: 'Chances of Bitcoin reaching $100,000 in 2021 was 40%'
           Query: Bitcoin, price, cryptocurrency, $100,000, 2021
        Now, generate a query for the following prediction: (Only generate query and no additional text or explanation.)
        """
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prediction_query},
            ],
        )
        
        return completion.choices[0].message.content.strip()
    
    def fetch_news_articles(self, search_query: str) -> List[Dict]:
        """Fetch news articles related to the prediction."""
        encoded_keywords = re.sub(r'[^\w\s]', '', search_query).replace(' ', '+')
        
        news_url = (
            f"https://api.thenewsapi.com/v1/news/all?"
            f"api_token={self.news_api_token}"
            f"&search={encoded_keywords}"
            f"&search_fields=title,main_text,description,keywords"
            f"&language=en"
            f"&published_after=2024-01-01"
            f"&sort=relevance_score"
        )
        
        news_response = requests.get(news_url)
        if news_response.status_code == 200:
            news_data = news_response.json()
            return news_data.get("data", [])
        return []
    
    def analyze_verification(self, prediction_query: str, all_sources: List[Dict]) -> Dict:
        """Analyze the sources to determine if the prediction was accurate."""
        article_summaries = "\n".join(
            [f"Title: {src['title']}, Source: {src['source']}, Description: {src['description']}, Snippet: {src['snippet']}" for src in all_sources]
        )

        system_prompt = """
        You are an AI analyst verifying predictions for Polymarket, a prediction market where users bet on real-world outcomes. Your task is to classify claims as TRUE, FALSE, or UNCERTAIN **only when evidence is insufficient**.

        ### Rules:
        1. **Classification Criteria**:
        - `TRUE`: The news articles **conclusively confirm** the prediction happened (e.g., "Bill passed" → voting records show it passed).
        - `FALSE`: The news articles **conclusively disprove** the prediction (e.g., "Company will move HQ" → CEO denies it).
        - `UNCERTAIN`: **Only if** evidence is missing, conflicting, or outdated (e.g., no articles after the predicted event date).

        2. **Evidence Standards**:
        - Prioritize **recent articles** (within 7 days of prediction date).
        - Trust **primary sources** (government releases, official statements) over opinion pieces.
        - Ignore irrelevant or off-topic articles.

        3. **Conflict Handling**:
        - If sources conflict, weigh authoritative sources (e.g., Reuters) higher than fringe outlets.
        - If timing is unclear (e.g., "will happen next week" but no update), default to `UNCERTAIN`.
        
        """       

        analysis_prompt = f"""
        The prediction is: "{prediction_query}". 

        Here are some recent news articles about this topic:
        {article_summaries}

        Based on this data, determine if the prediction was accurate. 
        Summarize the key evidence and provide the output in **JSON format** with the following structure:

        {{
          "result": "TRUE/FALSE/UNCERTAIN",
          "summary": "Brief explanation of why the claim is classified as TRUE, FALSE, or UNCERTAIN based on the news articles."
        }}

        Ensure the response is **valid JSON** with no additional text.
        """
        
        ai_verification = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        
        match = re.search(r"\{(.*)\}", ai_verification.choices[0].message.content, re.DOTALL)
        if match:
            ai_verification_result = "{" + match.group(1) + "}"
            try:
                return json.loads(ai_verification_result)
            except json.JSONDecodeError:
                return {
                    "result": "UNCERTAIN",
                    "summary": "Could not analyze the prediction due to formatting issues."
                }
        else:
            return {
                "result": "UNCERTAIN",
                "summary": "Could not analyze the prediction due to formatting issues."
            }
    
    def verify_prediction(self, prediction_query: str) -> Dict:
        """Main method to verify a prediction."""
        # Generate search query
        search_query = self.generate_search_query(prediction_query)
        print(f"Generated Search Query: {search_query}")
        
        # Fetch news articles
        articles = self.fetch_news_articles(search_query)
        
        # Fetch Google search results
        google_results = self.fetch_google_results(prediction_query)
        
        # Prepare sources from both APIs
        all_sources = [
            {"title": a['title'], "source": a['source'], "published": a['published_at'], "description": a['description'], "snippet": a['snippet']} for a in articles
        ] + [
            {"title": g['title'], "source": g['link'], "snippet": g['snippet'], "description": g.get('pagemap', {}).get('metatags', [{}])[0].get('og:description', '') if 'pagemap' in g else "", "published": "N/A"} for g in google_results
        ]
        
        if not all_sources:
            return {
                "result": "UNCERTAIN",
                "summary": "No relevant information found to verify this prediction.",
                "sources": []
            }
        
        # Analyze verification
        verification_data = self.analyze_verification(prediction_query, all_sources)
        
        # Final result
        final_result = {
            "result": verification_data["result"],
            "summary": verification_data["summary"],
            "sources": all_sources
        }
        
        return final_result


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
    


# Update the initiate_chat message too
def run_prediction_analysis():
    # Create agents
    assistant, user_proxy = create_prediction_agents()
    # assistant.register_reply(user_proxy, assistant.reply)
    # user_proxy.register_reply(assistant, user_proxy.reply)
    # Start the conversation
    user_proxy.initiate_chat(
        assistant,
        message="""I'd like to analyze predictions. Please select one of these specific options:
1. Find predictions on [specific topic], also give account names of users who made them. Try your best to show atleast 10 predictions.
2. Build profile for [@predictor_handle]. Show me their all prediction history and analysis.
3. Verify prediction: "[exact prediction text]"
4. Calculate credibility score for [@predictor_handle]

Which option would you like to proceed with? (1-4)"""
    )   
    print("\n=== FULL CONVERSATION HISTORY ===")
    for message in user_proxy.chat_messages[assistant]:
        print(f"{message['role']}: {message['content']}")
    
    # Check which functions were called
    print("\n=== FUNCTION EXECUTION SUMMARY ===")
    if hasattr(user_proxy, 'function_executions'):
        for execution in user_proxy.function_executions:
            print(f"Called {execution['name']} with args: {execution['args']}")
    else:
        print("No functions were executed")

# ============ MAIN FUNCTION ============"""
#if __name__ == "__main__":
    # Example 1: Find predictions on a topic
    #prediction_finder = PredictionFinder(groq_client, DATURA_API_KEY, DATURA_API_URL)
    # topic_predictions = prediction_finder.find_predictions("Next Prime Minister of Canada after elections?")
    # print(json.dumps(topic_predictions, indent=2))
    
    # Example 2: Build a predictor profile
    #predictor_profiler = PredictorProfiler(groq_client, DATURA_API_KEY, DATURA_API_URL)
    # profile = predictor_profiler.build_profile("Polymarket")
    # print(json.dumps(profile, indent=2))
    
    # Example 3: Verify a prediction
    #prediction_verifier = PredictionVerifier(groq_client, NEWS_API_TOKEN, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    # verification = prediction_verifier.verify_prediction("Chances of France winning the 2022 FIFA World Cup Final is 60%")
    # print(json.dumps(verification, indent=2))
    
    # NEW EXAMPLE: Calculate credibility score for a user
    #credibility_score = predictor_profiler.calculate_credibility_score("mdt546", prediction_verifier)
    # print("Credibility Score Analysis:")
    # print(json.dumps(credibility_score, indent=2))
    # # Store the credibility score in a JSON file
    # with open("credibility_score.json", "w") as file:
    #     json.dump(credibility_score, file, indent=2)
    # Run the autogen chat interface
    #run_prediction_analysis()