<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# Design Specialized Agents for a Recommendation System


Estimated time needed: **45** minutes


## Learning Objectives
After completing this lab, you'll be able to:

- Design agent goals, backstories, and personas using LangChain
- Create agent instances with clear responsibilities and expertise
- Design tasks for each agent with structured descriptions and expected outputs
- Apply ReAct and few-shot prompting patterns to agent design


## Introduction

In this lab, you will design a multi-agent system for restaurant and recipe recommendations. You will define six specialized agents, each with a distinct role in the recommendation pipeline:

1. **User Profile Generator**: Analyzes user data to extract preferences and patterns
2. **RAG Retriever**: Queries vector databases for relevant restaurants and recipes
3. **Food Trend Analyst**: Identifies current food trends and popular ingredients
4. **Food Style Expert**: Analyzes cuisines, cooking methods, and flavor profiles
5. **Nutrition Expert**: Evaluates nutritional content and dietary restrictions
6. **Recommendation Expert**: Synthesizes all insights into final recommendations

By breaking down the recommendation task into specialized roles, you create a modular, maintainable system where each agent focuses on what it does best.

**Note:** Throughout this lab, you'll answer strategically placed questions designed to reinforce your learning and complete the checklist.


## Table of contents

<font size = 3>    

1. [Install the required libraries](#Install-the-required-libraries)
2. [Import required libraries](#Import-required-libraries)
3. [Configure API access](#Configure-API-access)
4. [Define agent roles, goals, and backstories](#Define-agent-roles,-goals,-and-backstories)
   1. [Agent 1: User Profile Generator](#Agent-1:-User-Profile-Generator)
   2. [Agent 2: RAG Retriever](#Agent-2:-RAG-Retriever)
   3. [Agent 3: Food Trend Analyst](#Agent-3:-Food-Trend-Analyst)
   4. [Agent 4: Food Style Expert](#Agent-4:-Food-Style-Expert)
   5. [Agent 5: Nutrition Expert](#Agent-5:-Nutrition-Expert)
   6. [Agent 6: Recommendation Expert](#Agent-6:-Recommendation-Expert)
5. [Design agent tasks](#Design-agent-tasks)
   1. [Task 1: Generate the user profile](#Task-1:-Generate-the-user-profile)
   2. [Task 2: Retrieve relevant restaurants and recipes](#Task-2:-Retrieve-relevant-restaurants-and-recipes)
   3. [Task 3: Analyze food trends](#Task-3:-Analyze-food-trends)
   4. [Task 4: Design the task for the Food Style Expert](#Task-4:-Design-the-task-for-the-Food-Style-Expert)
   5. [Task 5: Evaluate nutrition and dietary fit](#Task-5:-Evaluate-nutrition-and-dietary-fit)
   6. [Task 6: Generate final recommendations](#Task-6:-Generate-final-recommendations)
6. [Test individual agents](#Test-individual-agents)
7. [Summary of agents and tasks](#Summary-of-agents-and-tasks)
</font>
</div>


## **Screenshot requirement for this lab**

You will be prompted to take a screenshot and save it on your own device. You will need this screenshot either to answer graded quiz questions or to upload as your submission for the Final Project at the end of this course. You can use various free screen-grabbing tools or your operating system's shortcut keys to do this (for example, `Alt+PrintScreen` on Windows and `Command+shift+4` on Mac).
**Note**: The screenshot can be saved with either the **.jpg** or **.png** extension.


### Install the required libraries

All the required libraries are __not__ pre-installed in the Skills Network Labs environment. __You need to run the following cell__ to install them. This might take a few minutes.


----



```python
%%capture
%pip install openai==1.99.9 | tail -n 1
%pip install langchain==0.3.0 | tail -n 1
%pip install langchain-community==0.3.27 | tail -n 1 
%pip install langchain-core==0.3.74 | tail -n 1 
%pip install langchain_experimental==0.3.4 | tail -n 1 
%pip install openai==1.99.9 | tail -n 1
```

### Import required libraries

Begin by importing the libraries and packages you will need to complete this lab.



```python
import os                         
from typing import List, Dict, Any                           
from openai import OpenAI
from langchain.prompts import PromptTemplate                             
from langchain_core.messages import HumanMessage, SystemMessage
```

### Configure API access 

To use LangChain with OpenAI models, you need to configure your API key. For this lab, you'll use a placeholder approach. In production, you would set your actual OpenAI API key.

**Note**: If you have an OpenAI API key, you can set it here. Otherwise, you can proceed with the design phase without running the agents.



```python
# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the OpenAI client
client = OpenAI()

# Define the model to use
MODEL = "gpt-5"
```

## Define agent roles, goals, and backstories

Each agent needs three core components:
- **Role**: A concise title describing the agent's function
- **Goal**: A clear statement of what the agent aims to accomplish
- **Backstory**: A narrative that establishes expertise and personality

Let's define these for all six agents.


### Agent 1: User Profile Generator


The recommendation pipeline begins by transforming raw user activity into structured intelligence. This agent converts visit history and social media signals into a meaningful user profile that captures preferences, habits, and constraints, creating the foundation that every downstream agent relies on.



```python
user_profile_agent_config = {
    "role": "User Profile Generator",
    "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile including preferences, dietary restrictions, favorite cuisines, and dining patterns.",
    "backstory": """You are an expert user behavior analyst with 10 years of experience in the food and hospitality industry. 
    You excel at reading between the lines to understand not just what users say they like, but what their actions reveal 
    about their true preferences. You have a talent for identifying patterns in dining behavior, recognizing subtle preferences, 
    and building rich user profiles that capture both explicit and implicit food preferences. You understand that a user's 
    social media posts and check-ins tell a story about their culinary journey, and you're skilled at extracting meaningful 
    insights from unstructured data."""
}

print(f"Role: {user_profile_agent_config['role']}")
print(f"\nGoal: {user_profile_agent_config['goal']}")
print(f"\nBackstory: {user_profile_agent_config['backstory']}")
```

    Role: User Profile Generator
    
    Goal: Analyze user restaurant visit history and social media posts to create a comprehensive profile including preferences, dietary restrictions, favorite cuisines, and dining patterns.
    
    Backstory: You are an expert user behavior analyst with 10 years of experience in the food and hospitality industry. 
        You excel at reading between the lines to understand not just what users say they like, but what their actions reveal 
        about their true preferences. You have a talent for identifying patterns in dining behavior, recognizing subtle preferences, 
        and building rich user profiles that capture both explicit and implicit food preferences. You understand that a user's 
        social media posts and check-ins tell a story about their culinary journey, and you're skilled at extracting meaningful 
        insights from unstructured data.

### Agent 2: RAG Retriever


Once a structured profile is established, this agent expands the search into the broader knowledge space. It translates user preferences into semantic queries against multimodal vector databases, retrieving restaurants, recipes, and related content that closely align with the user’s interests.



```python
rag_retriever_agent_config = {
    "role": "RAG Retriever",
    "goal": "Query multimodal vector databases to retrieve relevant restaurants, recipes, and food-related content based on user profiles and similarity search.",
    "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search. 
    You understand how embeddings capture meaning and can craft queries that retrieve the most relevant information 
    from large collections of restaurant data, recipes, and food images. You know when to use similarity search versus 
    filtered search, and you can balance relevance with diversity to ensure recommendations aren't repetitive. 
    You've worked with Pinecone, Weaviate, and ChromaDB, and you understand the nuances of multimodal retrieval 
    where text and images work together to represent food experiences."""
}

print(f"Role: {rag_retriever_agent_config['role']}")
print(f"\nGoal: {rag_retriever_agent_config['goal']}")
print(f"\nBackstory: {rag_retriever_agent_config['backstory']}")
```

    Role: RAG Retriever
    
    Goal: Query multimodal vector databases to retrieve relevant restaurants, recipes, and food-related content based on user profiles and similarity search.
    
    Backstory: You are a data retrieval specialist with expertise in vector databases and semantic search. 
        You understand how embeddings capture meaning and can craft queries that retrieve the most relevant information 
        from large collections of restaurant data, recipes, and food images. You know when to use similarity search versus 
        filtered search, and you can balance relevance with diversity to ensure recommendations aren't repetitive. 
        You've worked with Pinecone, Weaviate, and ChromaDB, and you understand the nuances of multimodal retrieval 
        where text and images work together to represent food experiences.

### Agent 3: Food Trend Analyst


After relevant options are retrieved, this agent evaluates them through the lens of current culinary trends and cultural shifts. By identifying emerging ingredients, dining movements, and evolving food preferences, it ensures that recommendations are not only personalized, but also timely and socially relevant.



```python
food_trend_analyst_config = {
    "role": "Food Trend Analyst",
    "goal": "Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.",
    "backstory": """You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across 
    global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to 
    Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food 
    movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream. 
    You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a 
    lasting trend."""
}

print(f"Role: {food_trend_analyst_config['role']}")
print(f"\nGoal: {food_trend_analyst_config['goal']}")
print(f"\nBackstory: {food_trend_analyst_config['backstory']}")
```

    Role: Food Trend Analyst
    
    Goal: Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.
    
    Backstory: You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across 
        global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to 
        Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food 
        movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream. 
        You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a 
        lasting trend.

### Agent 4: Food Style Expert


### **Question 1**: Complete the Food Style Expert configuration below by filling in the `goal` field.

**Hint**: This agent should analyze cuisine types, regional variations, cooking methods, and flavor profiles to match user preferences with appropriate food styles.



```python
food_style_expert_config = {
    "role": "Food Style Expert",
    "goal": "Analyze cuisine types, regional variations, cooking methods, and flavor profiles to match user preferences with appropriate food styles.",
    "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines. 
    You've cooked in kitchens across five continents and understand the techniques, ingredients, and cultural contexts 
    that define different food traditions. You can distinguish Sichuan from Cantonese, Neapolitan pizza from Roman, 
    and Nashville hot chicken from Buffalo wings. You understand flavor profiles—umami-rich, bright and acidic, 
    rich and creamy—and can map them to user preferences. You respect culinary heritage while staying open to fusion 
    and innovation."""
}
```

Take a screenshot of the Python code, clearly showing your implementation from Question 1. Name the screenshot **M3L1_food_style_expert_goal.jpg** (or .png). The screenshot should show the Food Style Expert agent configuration with all three fields—role, goal, and backstory—present. This screenshot will be used as one of the submission components for your Final Project.


### Agent 5: Nutrition Expert


After trends and preferences are considered, this agent evaluates recommendations through a health and wellness lens. It ensures that suggested dishes align with dietary restrictions, nutritional goals, and allergen sensitivities, adding a critical layer of safety and balance to the decision-making process.



```python
nutrition_expert_config = {
    "role": "Nutrition Expert",
    "goal": "Evaluate nutritional content, identify allergens, assess dietary restrictions, and ensure recommendations align with users' health and wellness goals.",
    "backstory": """You are a registered dietitian with a master's degree in nutrition science and 8 years of clinical experience. 
    You understand macronutrients, micronutrients, and how different diets (keto, Mediterranean, plant-based, etc.) affect health. 
    You can quickly assess whether a dish fits within dietary restrictions like gluten-free, dairy-free, or low-sodium. 
    You're also sensitive to food allergies and intolerances, and you know how to balance health considerations with the 
    pleasure of eating. You believe that good nutrition doesn't mean sacrificing flavor or enjoyment."""
}

print(f"Role: {nutrition_expert_config['role']}")
print(f"\nGoal: {nutrition_expert_config['goal']}")
print(f"\nBackstory: {nutrition_expert_config['backstory']}")
```

    Role: Nutrition Expert
    
    Goal: Evaluate nutritional content, identify allergens, assess dietary restrictions, and ensure recommendations align with users' health and wellness goals.
    
    Backstory: You are a registered dietitian with a master's degree in nutrition science and 8 years of clinical experience. 
        You understand macronutrients, micronutrients, and how different diets (keto, Mediterranean, plant-based, etc.) affect health. 
        You can quickly assess whether a dish fits within dietary restrictions like gluten-free, dairy-free, or low-sodium. 
        You're also sensitive to food allergies and intolerances, and you know how to balance health considerations with the 
        pleasure of eating. You believe that good nutrition doesn't mean sacrificing flavor or enjoyment.

### Agent 6: Recommendation Expert


After each specialized agent has contributed its insights, this agent brings everything together into a final decision. It synthesizes user preferences, retrieved options, trend awareness, culinary analysis, and nutritional considerations into cohesive, well-balanced recommendations that are both personalized and compelling.



```python
recommendation_expert_config = {
    "role": "Recommendation Expert",
    "goal": "Synthesize insights from all agents—user profiles, retrieved data, trends, food styles, and nutrition—into cohesive, well-reasoned restaurant and recipe recommendations.",
    "backstory": """You are a recommendation systems architect with experience building personalization engines for major 
    food delivery platforms and recipe apps. You understand how to balance multiple signals—relevance, diversity, novelty, 
    and serendipity—to create recommendations that delight users. You know when to play it safe with familiar favorites 
    and when to suggest something unexpected. You're skilled at synthesizing complex, sometimes conflicting information 
    from multiple sources into clear, actionable recommendations. You write in a warm, engaging tone that makes users 
    excited to try new restaurants and recipes."""
}

print(f"Role: {recommendation_expert_config['role']}")
print(f"\nGoal: {recommendation_expert_config['goal']}")
print(f"\nBackstory: {recommendation_expert_config['backstory']}")
```

    Role: Recommendation Expert
    
    Goal: Synthesize insights from all agents—user profiles, retrieved data, trends, food styles, and nutrition—into cohesive, well-reasoned restaurant and recipe recommendations.
    
    Backstory: You are a recommendation systems architect with experience building personalization engines for major 
        food delivery platforms and recipe apps. You understand how to balance multiple signals—relevance, diversity, novelty, 
        and serendipity—to create recommendations that delight users. You know when to play it safe with familiar favorites 
        and when to suggest something unexpected. You're skilled at synthesizing complex, sometimes conflicting information 
        from multiple sources into clear, actionable recommendations. You write in a warm, engaging tone that makes users 
        excited to try new restaurants and recipes.

### **Question 2**: Which agent is responsible for querying the vector database built in Module 2?


#### **Answer**: You can use this cell to enter your answer.


## Create agent instances

Now that we've defined the roles, goals, and backstories, let's create agent instances. In LangChain, we'll use the ReAct pattern to create agents that can reason and act.

For this lab, we'll create simplified agent representations using system prompts.



```python
def create_agent_prompt(agent_config: Dict[str, str]) -> str:
    """Create a system prompt for an agent based on its configuration."""
    prompt = f"""You are a {agent_config['role']}.
    
Your goal: {agent_config['goal']}

Your background: {agent_config['backstory']}

Always respond in a professional, helpful manner that reflects your expertise.
"""
    return prompt
```


```python
# Create agent prompts
user_profile_prompt = create_agent_prompt(user_profile_agent_config)
rag_retriever_prompt = create_agent_prompt(rag_retriever_agent_config)
food_trend_prompt = create_agent_prompt(food_trend_analyst_config)
food_style_prompt = create_agent_prompt(food_style_expert_config)
nutrition_prompt = create_agent_prompt(nutrition_expert_config)
recommendation_prompt = create_agent_prompt(recommendation_expert_config)

print("Agent prompts created successfully!")
```

    Agent prompts created successfully!

Let's test one agent prompt to see how it looks:



```python
print("=" * 80)
print("FOOD TREND ANALYST PROMPT")
print("=" * 80)
print(food_trend_prompt)
```

    ================================================================================
    FOOD TREND ANALYST PROMPT
    ================================================================================
    You are a Food Trend Analyst.
    
    Your goal: Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.
    
    Your background: You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across 
        global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to 
        Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food 
        movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream. 
        You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a 
        lasting trend.
    
    Always respond in a professional, helpful manner that reflects your expertise.


## Design agent tasks


Each agent needs specific tasks that define what it should accomplish. A well-designed task includes:
- **Description**: What the agent should do
- **Expected output**: Format and content of the result
- **Agent**: Which agent performs this task


### Task 1: Generate the user profile

Create a task configuration for the User Profile Generator that analyzes the user's restaurant visit history and social media posts to create a comprehensive preference profile.



```python
task_generate_profile = {
    "description": """Analyze the user's restaurant visit history and social media posts to create a comprehensive profile.
    Extract the following information:
    - Favorite cuisines and cuisine categories
    - Dietary restrictions or preferences (vegetarian, vegan, gluten-free, etc.)
    - Preferred dining occasions (casual, fine dining, quick bites)
    - Price sensitivity
    - Adventurousness (comfort food lover vs. culinary explorer)
    - Flavor preferences (spicy, sweet, savory, etc.)
    - Frequency of dining out
    
    Provide specific examples from the user's history to support each insight.""",
    
    "expected_output": """A structured user profile in JSON format with keys: 
    favorite_cuisines, dietary_restrictions, dining_occasions, price_range, 
    adventurousness_score (1-10), flavor_preferences, dining_frequency.
    Include a summary paragraph explaining the user's dining personality.""",
    
    "agent": "User Profile Generator"
}

print(f"Task: {task_generate_profile['description'][:100]}...")
print(f"\nExpected Output: {task_generate_profile['expected_output'][:100]}...")
print(f"\nAgent: {task_generate_profile['agent']}")
```

    Task: Analyze the user's restaurant visit history and social media posts to create a comprehensive profile...
    
    Expected Output: A structured user profile in JSON format with keys: 
        favorite_cuisines, dietary_restrictions, di...
    
    Agent: User Profile Generator

### Task 2: Retrieve relevant restaurants and recipes

Create a task configuration for the RAG Retriever that queries the vector database to retrieve relevant restaurants and recipes based on the user profile.



```python
task_retrieve_candidates = {
    "description": """Based on the user profile, query the vector database to retrieve:
    - Top 20 restaurants that match the user's preferences
    - Top 20 recipes that align with their taste and dietary needs
    
    Use similarity search with the user's favorite cuisines and flavor preferences as the query.
    Apply filters for dietary restrictions, price range, and location (if provided).
    Ensure diversity in the results—don't retrieve 20 Italian restaurants if the user likes multiple cuisines.""",
    
    "expected_output": """Two lists in JSON format:
    - restaurants: Array of restaurant objects with fields: name, cuisine_type, price_range, rating, description
    - recipes: Array of recipe objects with fields: name, cuisine_type, difficulty, prep_time, ingredients, description""",
    
    "agent": "RAG Retriever"
}

print(f"Task: {task_retrieve_candidates['description'][:100]}...")
print(f"\nExpected Output: {task_retrieve_candidates['expected_output'][:100]}...")
print(f"\nAgent: {task_retrieve_candidates['agent']}")
```

    Task: Based on the user profile, query the vector database to retrieve:
        - Top 20 restaurants that matc...
    
    Expected Output: Two lists in JSON format:
        - restaurants: Array of restaurant objects with fields: name, cuisine_...
    
    Agent: RAG Retriever

### Task 3: Analyze food trends

Create a task configuration for the Food Trend Analyst that identifies current food trends and emerging culinary movements relevant to the retrieved restaurants and recipes.



```python
task_analyze_trends = {
    "description": """Analyze current food trends relevant to the retrieved restaurants and recipes.
    Identify:
    - Trending ingredients or techniques in the retrieved items
    - Popular dining concepts or restaurant types
    - Emerging culinary movements that align with the user's interests
    - Seasonal trends or timely food moments
    
    Provide context on why these trends matter and how they enhance the recommendations.""",
    
    "expected_output": """A trends analysis with:
    - List of 3-5 relevant trends with descriptions
    - Explanation of how each trend relates to the user's profile
    - Suggestions for which restaurants or recipes align with these trends""",
    
    "agent": "Food Trend Analyst"
}
```

### **Task 4**: Design the task for the Food Style Expert

Create a task configuration for the Food Style Expert that analyzes the cuisine types and flavor profiles of the retrieved restaurants and recipes.



```python
## Type your answer here

task_analyze_food_styles = {
    "description": """Analyze cuisine types, regional variations, cooking methods, and flavor profiles to match user preferences with appropriate food styles.
    Identify:
    - Cooking techniques used in the cuisine
    - Ingredients and cooking methods used
    - Regional variations and flavor profiles used
    - Potential user preferences based on the food""", 
    "expected_output": """A cuisine type and flavor profile analysis of the retrived restaurants and recipes with:
    - Identified cuisine type
    - Flavor analysis with at least one of the 5 main tastes in food (salty, sweet, bitter, umami, sour""",
    "agent": "Food Style Expert"
}
```

### Task 5: Evaluate nutrition and dietary fit

Create a task configuration for the Nutrition Expert that evaluates the nutritional content and dietary compatibility of the retrieved restaurants and recipes



```python
task_evaluate_nutrition = {
    "description": """Evaluate the nutritional aspects of the retrieved restaurants and recipes.
    Check for:
    - Alignment with dietary restrictions (vegetarian, vegan, gluten-free, etc.)
    - Potential allergens or ingredients to avoid
    - Nutritional balance (protein, vegetables, whole grains)
    - Healthfulness relative to the user's goals
    
    Flag any items that don't meet the user's dietary needs and explain why.""",
    
    "expected_output": """A nutrition evaluation with:
    - List of items that meet all dietary restrictions
    - Items flagged for potential concerns (allergens, restrictions)
    - Nutritional highlights (high protein, vegetable-rich, etc.)
    - Overall assessment of how well the options support the user's health goals""",
    
    "agent": "Nutrition Expert"
}
```

### Task 6: Generate final recommendations

Create a task configuration for the Recommendation Expert that synthesizes insights from all previous agents into final restaurant and recipe recommendations.



```python
task_generate_recommendations = {
    "description": """Synthesize insights from all previous agents to generate final recommendations.
    Create:
    - Top 5 restaurant recommendations with detailed explanations
    - Top 5 recipe recommendations with detailed explanations
    
    For each recommendation, explain:
    - Why it matches the user's profile
    - How it aligns with current trends (if applicable)
    - What makes it a great fit in terms of food style
    - Any nutritional benefits or considerations
    
    Write in an engaging, enthusiastic tone that makes the user excited to try these options.""",
    
    "expected_output": """A recommendations report with:
    - restaurants: Array of 5 restaurant recommendations with name, description, and detailed reasoning
    - recipes: Array of 5 recipe recommendations with name, description, and detailed reasoning
    - Each recommendation should include a personalized explanation (2-3 sentences) of why it's a great match""",
    
    "agent": "Recommendation Expert"
}

print(f"Task: {task_generate_recommendations['description'][:100]}...")
print(f"\nExpected Output: {task_generate_recommendations['expected_output'][:100]}...")
print(f"\nAgent: {task_generate_recommendations['agent']}")
```

    Task: Synthesize insights from all previous agents to generate final recommendations.
        Create:
        - To...
    
    Expected Output: A recommendations report with:
        - restaurants: Array of 5 restaurant recommendations with name, d...
    
    Agent: Recommendation Expert

### **Question 3**: How many tasks have we defined in total for the multi-agent system?


#### **Answer**: You can use this cell to enter your answer.


## Test individual agents


Let's test one agent to ensure it responds appropriately. We'll test the User Profile Generator with a sample user input.



```python
# Sample user data
sample_user_data = """
Restaurant Visit History:
- Visited "Spice Route" (Indian, $$) 5 times in the last 3 months
- Visited "Green Earth Cafe" (Vegan, $) 3 times
- Visited "Ramen House" (Japanese, $$) 2 times
- Visited "Taco Fiesta" (Mexican, $) 4 times

Social Media Posts:
- "Loving this spicy curry at Spice Route! 🌶️🔥"
- "Trying to eat more plant-based meals. This vegan bowl is delicious!"
- "Best ramen I've had in ages. The broth is perfection."
- "Late night tacos are the best tacos 🌮"
"""
```


```python
def test_agent(agent_prompt: str, user_input: str) -> str:
    """Test an agent by sending it a sample input."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": agent_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content
```


```python
# Test the User Profile Generator
# Note: This will only work if you have set your OpenAI API key
try:
    profile_result = test_agent(
        agent_prompt=user_profile_prompt,
        user_input=f"Analyze this user data and create a profile:\n\n{sample_user_data}"
    )
    print("=" * 80)
    print("USER PROFILE GENERATED")
    print("=" * 80)
    print(profile_result)
except Exception as e:
    print(f"Note: Agent testing requires a valid OpenAI API key. Error: {e}")
    print("\nYou have successfully designed all agents and tasks.")
    print("In Lesson 2, you will integrate these agents into a working system.")
```

    ================================================================================
    USER PROFILE GENERATED
    ================================================================================
    User Profile Summary
    - A value-conscious, casual diner who eats out ~4–5 times per month, favoring bold, spicy, and comfort-forward foods. Strong loyalty to Indian cuisine and late-night Mexican, with a growing interest in plant-based options and a clear appreciation for high-quality broths and well-executed staples.
    
    Dietary Profile
    - Diet type: Omnivore with plant-forward intent (“Trying to eat more plant-based meals” + 3 vegan visits)
    - Restrictions: None indicated. Comfortable with gluten (ramen, tacos) and likely fine with moderate dairy (common in Indian curries), but not confirmed.
    - Spice tolerance: High (explicit enjoyment of “spicy curry” with chili/heat emojis)
    
    Favorite Cuisines (ranked by behavior + sentiment)
    1) Indian ($$) — Highest loyalty (5 visits/3 months) and enthusiastic sentiment about spicy curry
    2) Mexican ($) — Frequent (4 visits), late-night appeal, casual/snackable comfort
    3) Vegan/Plant-based ($) — Intent-driven and recurring (3 visits), suggests health/sustainability interest
    4) Japanese/Ramen ($$) — Quality-seeking (“broth is perfection”); comfort and umami focus (2 visits)
    
    Flavor and Dish Preferences
    - Loves chili heat, aromatic spices, and robust flavors
    - Drawn to umami-rich, slow-crafted items (ramen broth, curries)
    - Enjoys casual, handheld/street-food formats (tacos)
    - Positive toward vegetable-forward bowls; open to meatless mains without feeling deprived
    
    Price Sensitivity and Occasion
    - Comfortable in the $–$$ range; no evidence of $$$ splurges
    - Mix of quick, affordable bites (Vegan café, tacos) and mid-priced sit-down meals (Indian, ramen)
    
    Dining Patterns
    - Frequency: ~14 visits over 3 months (~1.1 visits/week)
    - Timing: Evidence of late-night dining (tacos); otherwise likely a mix of casual lunches/dinners
    - Loyalty vs variety: Repeats favorites (notably Indian), but experiments within a trusted set of venues
    - Social signals: Shares when food is notably spicy, plant-based success, or high craftsmanship (broth)
    
    Implicit Insights (“read between the lines”)
    - Heat-seeker: Will likely choose “spicy” versions and appreciate heat-level customization
    - Comfort + Craft: Values dishes where technique shines (broth, curry), not just novelty
    - Plant-forward momentum: Not strict vegan/vegetarian, but receptive to compelling meatless options, especially if bold/spicy
    - Late-night cravings: Mexican fits a convenience/comfort niche after hours
    
    Engagement Opportunities
    - Highlight spicy specials and heat-scale options (vindaloo, spicy chana/eggplant, extra-chili salsas, tantanmen-style ramen)
    - Offer plant-based versions of favorites: vegan curries, tofu/tempeh ramen, veggie/jackfruit tacos with robust salsas
    - Late-night value bundles for tacos or spicy small plates
    - Loyalty nudges at top spots (Indian/Mexican) and discovery offers at quality-forward ramen/vegan venues
    
    Confidence Levels
    - High: Preference for spicy food; strong affinity for Indian; enjoyment of tacos; comfort with $–$$
    - Medium: Late-night dining as a recurring behavior; sustained plant-forward shift
    - Low: Specific allergens or strict dietary restrictions (none stated)

### **Question 4**: Based on the sample user data, which dietary preference is the user exploring?


#### **Answer**: You can use this cell to enter your answer.


## Summary of agents and tasks



```python
# Create a summary of all agents and their tasks
agents_summary = [
    {"agent": "User Profile Generator", "task": "Generate User Profile"},
    {"agent": "RAG Retriever", "task": "Retrieve Relevant Restaurants and Recipes"},
    {"agent": "Food Trend Analyst", "task": "Analyze Food Trends"},
    {"agent": "Food Style Expert", "task": "Analyze Food Styles"},
    {"agent": "Nutrition Expert", "task": "Evaluate Nutrition and Dietary Fit"},
    {"agent": "Recommendation Expert", "task": "Generate Final Recommendations"}
]

print("=" * 80)
print("MULTI-AGENT SYSTEM SUMMARY")
print("=" * 80)
for i, item in enumerate(agents_summary, 1):
    print(f"{i}. {item['agent']:30} → {item['task']}")
```

    ================================================================================
    MULTI-AGENT SYSTEM SUMMARY
    ================================================================================
    1. User Profile Generator         → Generate User Profile
    2. RAG Retriever                  → Retrieve Relevant Restaurants and Recipes
    3. Food Trend Analyst             → Analyze Food Trends
    4. Food Style Expert              → Analyze Food Styles
    5. Nutrition Expert               → Evaluate Nutrition and Dietary Fit
    6. Recommendation Expert          → Generate Final Recommendations

### Congratulations!
You have successfully completed this lab.


### Summary

In this lab, you defined six specialized agents for a multi-agent recommendation system:
1. **User Profile Generator**: Analyzes user data to extract preferences
2. **RAG Retriever**: Queries vector databases for relevant content
3. **Food Trend Analyst**: Identifies current food trends
4. **Food Style Expert**: Analyzes cuisines and flavor profiles
5. **Nutrition Expert**: Evaluates nutritional content and dietary fit
6. **Recommendation Expert**: Synthesizes insights into final recommendations

You designed roles, goals, and backstories for each agent, and created detailed task specifications with expected outputs. In the next lesson, you will integrate these agents into a coordinated workflow and test the complete system.


## Authors


[Tenzin Migmar](https://author.skills.network/instructors/tenzin_migmar)


©IBM Corporation. All rights reserved.


<!--## Change Log
|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2026-02-03|1|Jianping Ye|Create lab|
|2026-02-10|2|Jojy John|ID Reviewed|
|2026-02-18|2|Prashant Juyal|QA edits|-->

