<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# Implement and Test a Multi-Agent Recommendation System


<h5>Estimated time: 45 minutes</h5>


## Learning Objectives
After completing this lab, you'll be able to:

- Design and implement a hybrid workflow for coordinating multiple agents
- Use LangGraph to build stateful multi-agent systems
- Create node functions that update shared state
- Build workflow graphs with sequential and parallel execution patterns
- Test multi-agent systems with diverse user inputs
- Evaluate recommendation quality and agent coordination


## Introduction

Earlier, you designed six specialized agents. In this lab, you will integrate them into a coordinated workflow that generates restaurant and recipe recommendations.

You will implement a **hybrid workflow** with four phases:
1. **User Analysis (Sequential)**: Generate a user profile
2. **Data Retrieval (Sequential)**: Query the vector database
3. **Analysis (Parallel)**: Analyze trends, food styles, and nutrition simultaneously
4. **Synthesis (Sequential)**: Generate final recommendations

By the end of this lab, you will have a working multi-agent system that you can test with a variety of user inputs.

**Note:** Throughout this lab, you'll answer strategically placed questions designed to reinforce your learning and complete the checklist.


## Table of contents

<font size = 3>    
    
1. [Install the required libraries](#Install-the-required-libraries)
2. [Import required libraries](#Import-required-libraries)
3. [Import agent configurations](#Import-agent-configurations)
4. [Define the shared state structure](#Define-the-shared-state-structure)
5. [Question 1: How many fields are in the AgentState structure?](#Question-1:-How-many-fields-are-in-the-AgentState-structure?)
6. [Create the node functions](#Create-the-node-functions)
7. [Helper function: Call agent](#Helper-function:-Call-agent)
8. [Node 1: Generate the user profile](#Node-1:-Generate-the-user-profile)
9. [Node 2: Retrieve the candidates](#Node-2:-Retrieve-the-candidates)
10. [Node 3: Analyze trends](#Node-3:-Analyze-trends)
11. [Node 4: Analyze food styles](#Node-4:-Analyze-food-styles)
     1. [Task 1: Complete the node_analyze_styles function](#Task-1:-Complete-the-node_analyze_styles-function)
12. [Node 5: Evaluate nutrition](#Node-5:-Evaluate-nutrition)
13. [Node 6: Generate recommendations](#Node-6:-Generate-recommendations)
14. [Question 2: Which three nodes execute in parallel during Phase 3?](#Question-2:-Which-three-nodes-execute-in-parallel-during-Phase-3?)
15. [Build the workflow graph](#Build-the-workflow-graph)
16. [Test the multi-agent system](#Test-the-multi-agent-system)
     1. [Test Case 1: The health-conscious user](#Test-Case-1:-The-health-conscious-user)
     2. [Test Case 2: The adventurous foodie](#Test-Case-2:-The-adventurous-foodie)
17. [Question 3: Based on the workflow design, which phase takes the longest time to execute and why?](#Question-3:-Based-on-the-workflow-design,-which-phase-takes-the-longest-time-to-execute-and-why?)
18. [Evaluate the recommendations](#Evaluate-the-recommendations)
19. [Question 4: What is the purpose of the workflow_step field in the AgentState?](#Question-4:-What-is-the-purpose-of-the-workflow_step-field-in-the-AgentState?)

</font>
</div>


## **Screenshot requirement for this lab**

You will be prompted to take a screenshot and save it on your own device. You will need this screenshot either to answer graded quiz questions or to upload as your submission for the Final Project at the end of this course. You can use various free screen-grabbing tools or your operating system's shortcut keys to do this (for example, `Alt+PrintScreen` on Windows and `Command+shift+4` on Mac).
**Note**: The screenshot can be saved with either the **.jpg** or **.png** extension.


----


### Install the required libraries


All the required libraries are __not__ pre-installed in the Skills Network Labs environment. __You need to run the following cell__ to install them, and this might take a few minutes.



```python
%%capture
%pip install openai==1.99.9 | tail -n 1
%pip install langchain==0.3.0
%pip install langchain-openai==0.2.0
%pip install langchain-community==0.3.0
%pip install langgraph==0.2.0
```

### Import required libraries

Let's import the libraries needed for building the multi-agent workflow.



```python
import os
import json
from typing import TypedDict, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
```


```python
# Configure OpenAI API
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

client = OpenAI()
MODEL = "gpt-5"
```

### Import agent configurations

Let's bring in the agent configurations you've worked on earlier.



```python
# Agent configurations from Lesson 1
agent_configs = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": """You are an expert user behavior analyst with 10 years of experience in the food industry. 
        You excel at identifying patterns in dining behavior and building rich user profiles."""
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search."""
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": """You are a culinary journalist who has spent 15 years covering food culture across global markets."""
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines."""
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": """You are a registered dietitian with 8 years of clinical experience."""
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final recommendations.",
        "backstory": """You are a recommendation systems architect with experience in personalization engines."""
    }
}

print("Agent configurations loaded successfully!")
```

    Agent configurations loaded successfully!

## Define the shared state structure

The shared state flows through the entire workflow. Each agent reads from it and updates it with their outputs.

You'll use a TypedDict to define the state structure.



```python
# Define the shared state structure as a dictionary.
# Every node reads from and writes to this state.
INITIAL_STATE = {
    # Input
    "user_input": "",
    
    # Phase 1: User Analysis
    "user_profile": {},
    
    # Phase 2: Data Retrieval
    "retrieved_restaurants": [],
    "retrieved_recipes": [],
    
    # Phase 3: Analysis (Parallel)
    "trend_analysis": {},
    "style_analysis": {},
    "nutrition_analysis": {},
    
    # Phase 4: Synthesis
    "final_recommendations": {},
    
    # Metadata
    "workflow_step": "start"
}

print(f"State structure defined with {len(INITIAL_STATE)} fields:")
for key in INITIAL_STATE:
    print(f"  - {key}")
```

    State structure defined with 9 fields:
      - user_input
      - user_profile
      - retrieved_restaurants
      - retrieved_recipes
      - trend_analysis
      - style_analysis
      - nutrition_analysis
      - final_recommendations
      - workflow_step

### **Question 1**: How many fields are in the AgentState structure?


#### **Answer**: You can use this cell to enter your answer.


## Create the node functions

Each node represents an agent's task. A node function:
1. Receives the current state
2. Performs the agent's task
3. Returns an updated state

Let's create node functions for all six agents.


### Helper function: Call agent



```python
def call_agent(agent_key: str, user_message: str) -> str:
    """Call an agent with a specific message and return its response."""
    config = agent_configs[agent_key]
    
    system_prompt = f"""You are a {config['role']}.
    
Your goal: {config['goal']}

Your background: {config['backstory']}

Respond with structured, actionable output."""
    
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content
```

### Node 1: Generate the user profile



```python
def node_generate_profile(state: dict) -> dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")
    
    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    
    try:
        response = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"✓ User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"⚠ Error generating profile: {e}")
        user_profile = {"error": str(e)}
    
    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state
```

### Node 2: Retrieve the candidates



```python
def node_retrieve_candidates(state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")
    
    profile = state["user_profile"]
    
    user_message = f"""Based on this user profile:
{json.dumps(profile, indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    
    try:
        response = call_agent("rag_retriever", user_message)
        retrieved_data = json.loads(response)
        restaurants = retrieved_data.get("restaurants", [])
        recipes = retrieved_data.get("recipes", [])
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []
    
    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state
```

### Node 3: Analyze trends



```python
def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    
    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}
    
    state["trend_analysis"] = trend_analysis
    return state

```

### Node 4: Analyze food styles


### **Task 1**: Complete the `node_analyze_styles` function

Fill in the `user_message` to instruct the Food Style Expert to analyze cuisine types and flavor profiles.



```python
def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Analyze cuisine types and flavor profiles in these options, and evaluate how well they match the user's preferences.

User Profile: {json.dumps(profile, indent=2)}

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

For each option, consider cuisine type, dominant flavors, and how closely it aligns with the user's favorite cuisines and flavor preferences.

Return ONLY valid JSON (no markdown, no extra text) in this format:
{{
  "cuisine_breakdown": [{{"cuisine": str, "key_characteristics": str}}],
  "flavor_profile": {{"dominant_flavors": [str], "spice_level": str}},
  "user_alignment": [{{"name": str, "type": "restaurant" or "recipe", "match_score": int (1-10), "reasoning": str}}],
  "summary": str
}}
"""
    
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}
    
    state["style_analysis"] = style_analysis
    return state
```

Take a screenshot of the Python code, clearly showing your implementation from Task 1. Name the screenshot ```M3L2_node_analyze_styles.jpg```.


### Node 5: Evaluate nutrition



```python
def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}
    
    state["nutrition_analysis"] = nutrition_analysis
    return state
```

### Node 6: Generate recommendations



```python
def node_generate_recommendations(state: dict) -> dict:
    """Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")
    
    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes": [{{"name": str, "reasoning": str}}]
}}

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"✓ Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recommendations = {"error": str(e)}
    
    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state
```

### **Question 2**: Which three nodes execute in parallel during Phase 3?


#### **Answer**: You can use this cell to enter your answer.


## Build the workflow graph

Now you'll build the workflow that connects all nodes. You'll use Python's `ThreadPoolExecutor` to run the Phase 3 analysis agents in parallel, mirroring what a framework like LangGraph would do under the hood.



```python
def run_workflow(user_input: str) -> dict:
    """Run the full multi-agent workflow.
    
    Phases:
      1. User Analysis      (sequential)
      2. Data Retrieval      (sequential)
      3. Analysis            (parallel – trends, styles, nutrition)
      4. Synthesis           (sequential)
    """
    
    # Initialize shared state
    state = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start"
    }
    
    # Phase 1 – Sequential
    state = node_generate_profile(state)
    
    # Phase 2 – Sequential
    state = node_retrieve_candidates(state)
    
    # Phase 3 – Parallel using ThreadPoolExecutor
    print("\n[Phase 3] Running analysis agents in parallel...")
    
    # Each function needs its own copy of state to read from,
    # and we merge their outputs back afterwards.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends   = executor.submit(node_analyze_trends, dict(state))
        future_styles   = executor.submit(node_analyze_styles, dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))
        
        result_trends   = future_trends.result()
        result_styles   = future_styles.result()
        result_nutrition = future_nutrition.result()
        
    # Merge parallel results back into state
    state["trend_analysis"]    = result_trends["trend_analysis"]
    state["style_analysis"]    = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]
    
    # Phase 4 – Sequential
    state = node_generate_recommendations(state)
    
    return state

print("✓ Workflow function built successfully!")
```

    ✓ Workflow function built successfully!

## Test the multi-agent system

Next, test the system with different user personas.

### Test Case 1: The health-conscious user



```python
test_user_1 = """
Restaurant Visit History:
- Visited "Green Bowl" (Vegan, $$) 8 times
- Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
- Visited "Juice Lab" (Smoothies, $) 3 times

Social Media Posts:
- "Loving my plant-based journey! 🌱"
- "This gluten-free Mediterranean bowl is amazing!"
- "Fresh juice is the best way to start the day."

Dietary Restrictions: Vegan, Gluten-Free
"""

print("="*80)
print("TEST CASE 1: Health-Conscious User")
print("="*80)

try:
    result_1 = run_workflow(test_user_1)
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(json.dumps(result_1["final_recommendations"], indent=2))
except Exception as e:
    print(f"\nTest requires valid OpenAI API key. Error: {e}")
```

    ================================================================================
    TEST CASE 1: Health-Conscious User
    ================================================================================
    
    [Phase 1] Generating user profile...✓ User profile generated: Health-conscious vegan and gluten-free diner who favors plant-based bowls and Mediterranean flavors, starts mornings with fresh juice, and prefers mid-priced casual spots. Shows moderate variety within dietary boundaries, leaning toward fresh, herbaceous, and tangy profiles.
    
    [Phase 2] Retrieving candidates from vector database...✓ Retrieved 20 restaurants and 20 recipes
    
    [Phase 3] Running analysis agents in parallel...
    
    [Phase 3a] Analyzing food trends...
    
    [Phase 3b] Analyzing food styles...
    
    [Phase 3c] Evaluating nutrition...✓ Identified 5 trends✓ Style analysis completed✓ Nutrition evaluation completed
    
    [Phase 4] Generating final recommendations...✓ Generated 5 restaurant recommendations
    ✓ Generated 5 recipe recommendations
    
    ================================================================================
    FINAL RECOMMENDATIONS
    ================================================================================
    {
      "restaurants": [
        {
          "name": "Zest & Zaatar",
          "reasoning": "All-vegan Mediterranean with a dedicated gluten-free falafel fryer hits your vegan/GF needs while keeping flavors bright with lemon, sumac, herbs, and garlicky tahini. It\u2019s ideal for quick weekday lunches or casual dinners at a comfortable $$ price."
        },
        {
          "name": "Citrus + Seed Juice Co.",
          "reasoning": "Perfect for breakfast or post-workout, their cold-pressed juices, ginger\u2013citrus shots, and smoothie bowls are fruit-forward and clean with certified gluten-free toppings. The citrusy, minty profiles align with your light, fresh start to the day."
        },
        {
          "name": "Olive & Thyme Plant Kitchen",
          "reasoning": "All-vegan Mediterranean plates feature lemon\u2013tahini grilled veg, quinoa tabbouleh (no bulgur), and clearly labeled GF options for stress-free ordering. Herbaceous, tangy flavors stay light while tofu skewers add satisfying plant protein."
        },
        {
          "name": "Green Fig Kitchen",
          "reasoning": "Israeli-inspired vegan bowls on gluten-free rice or millet bases bring citrus, herbs, roasted eggplant, and optional amba/schug for adjustable heat. It\u2019s a fresh, clean fit for weekday lunch or casual dinner within your $$ range."
        },
        {
          "name": "Cedar & Chickpea",
          "reasoning": "Lebanese vegan favorites like GF-friendly fattoush (with chickpea crisps), mujadara, and bold garlic\u2013tahini deliver herbaceous, tangy depth without heaviness. Bright sumac vinaigrettes and plenty of fresh herbs match your flavor profile."
        }
      ],
      "recipes": [
        {
          "name": "Lemon-Tahini Quinoa Tabbouleh",
          "reasoning": "Quinoa keeps it naturally gluten-free while parsley, mint, and lemon\u2013garlic tahini deliver the herbaceous, tangy profile you love. It\u2019s a 20-minute, light lunch staple with clean, fresh flavors."
        },
        {
          "name": "Chickpea \u2018Shawarma\u2019 Lettuce Wraps",
          "reasoning": "Spiced roasted chickpeas, pickles, and a lemon\u2013tahini drizzle give crisp, tangy bites that are totally vegan and GF. Ready in about 25 minutes, it\u2019s perfect for quick weekday lunches."
        },
        {
          "name": "Mediterranean Lentil Salad with Preserved Lemon",
          "reasoning": "Lentils, olives, dill, and chopped preserved lemon create a bright, satisfying salad that stays light and gluten-free. It comes together fast and aligns with your herbaceous, citrus-forward preferences."
        },
        {
          "name": "Herbaceous Baked Falafel with Green Tahini",
          "reasoning": "Baked (not fried) falafel with chickpea flour keeps it GF and lighter, while green tahini packs parsley\u2013cilantro\u2013garlic freshness. Great for meal-prep bowls or a casual dinner with crisp veggies."
        },
        {
          "name": "Grapefruit-Mint Green Juice",
          "reasoning": "Zesty grapefruit, cucumber, celery, green apple, spinach, and mint deliver a refreshing, citrus-forward morning boost. It\u2019s clean, hydrating, and perfectly aligned with your breakfast juice routine."
        }
      ]
    }

### Test Case 2: The adventurous foodie



```python
test_user_2 = """
Restaurant Visit History:
- Visited "Omakase Sushi" (Japanese Fine Dining, $$$$) 4 times
- Visited "Street Food Market" (International Fusion, $$) 6 times
- Visited "Molecular Gastronomy Lab" (Experimental, $$$$) 2 times

Social Media Posts:
- "Mind-blown by the 12-course tasting menu! 🤯"
- "Trying crickets for the first time. Surprisingly good!"
- "This molecular take on traditional ramen is art."

Dietary Restrictions: None
"""

print("="*80)
print("TEST CASE 2: Adventurous Foodie")
print("="*80)

try:
    result_2 = run_workflow(test_user_2)
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(json.dumps(result_2["final_recommendations"], indent=2))
except Exception as e:
    print(f"\nTest requires valid OpenAI API key. Error: {e}")
```

    ================================================================================
    TEST CASE 2: Adventurous Foodie
    ================================================================================
    
    [Phase 1] Generating user profile...✓ User profile generated: Balances frequent $$ street-food/fusion visits with regular $$$$ omakase and modernist tastings. Highly open to novelty (e.g., crickets) and drawn to multi-course chef-led experiences. Prefers Japanese and fusion with strong umami and seafood, enjoys textural creativity, and comfortably shifts between casual exploration and premium splurges.
    
    [Phase 2] Retrieving candidates from vector database...✓ Retrieved 20 restaurants and 20 recipes
    
    [Phase 3] Running analysis agents in parallel...
    
    [Phase 3a] Analyzing food trends...
    
    [Phase 3b] Analyzing food styles...
    
    [Phase 3c] Evaluating nutrition...✓ Identified 5 trends✓ Style analysis completed✓ Nutrition evaluation completed
    
    [Phase 4] Generating final recommendations...✓ Generated 5 restaurant recommendations
    ✓ Generated 5 recipe recommendations
    
    ================================================================================
    FINAL RECOMMENDATIONS
    ================================================================================
    {
      "restaurants": [
        {
          "name": "SingleThread",
          "reasoning": "Farm-driven Japanese-Californian kaiseki hits your love for chef-led progressions, pristine seafood, and hyper-seasonality. Expect layered dashi/koji umami, clean precision, and elegant texture choreography\u2014perfect for a $$$$ special-occasion splurge."
        },
        {
          "name": "Den",
          "reasoning": "Playful modern kaiseki delivers deep dashi flavors and inventive textures that reward high adventurousness. It blends ceremony with surprise, aligning with your desire for narrative, novelty, and chef-curated tasting experiences."
        },
        {
          "name": "Sushi Saito",
          "reasoning": "Benchmark Edo-mae omakase where rice seasoning, fish aging, and temperature control create ultra-clean, oceanic umami. Ideal for a focused, texture- and detail-obsessed counter experience."
        },
        {
          "name": "Sushi Noz",
          "reasoning": "Luxurious Edo-mae emphasizing controlled aging and seasoning to amplify pure, briny sweetness and precise textures. A refined $$$$ omakase that scratches your modernist curiosity without sacrificing cleanliness of flavor."
        },
        {
          "name": "n/naka",
          "reasoning": "Modern kaiseki marrying Japanese technique with California produce for an elegant, balanced progression. Pristine seafood, restrained seasoning, and seasonal storytelling fit your special-occasion tastings and preference for clean, umami-rich dishes."
        }
      ],
      "recipes": [
        {
          "name": "Crispy Nori Tacos with Tuna Tartare and Yuzu Kosho",
          "reasoning": "A street-food fusion bite that nails crunch-meets-silk textures and citrus-chili brightness. It\u2019s quick, playful, and seafood-forward\u2014perfect for casual discovery with friends."
        },
        {
          "name": "Chawanmushi with Uni and Snow Crab",
          "reasoning": "Silky custard showcases precise technique and clean, oceanic umami. Uni and crab add luxe sweetness and texture contrast for an elegant, tasting-menu feel at home."
        },
        {
          "name": "Shio Koji Black Cod with Charred Negi",
          "reasoning": "Koji tenderizes and layers savoriness while charred negi adds gentle smoke and bite. It\u2019s a pristine, seafood-forward dish that highlights your love of umami stacking and clean precision."
        },
        {
          "name": "Spherified Tomato Dashi Pearls with Burrata",
          "reasoning": "Modernist \u2018caviar\u2019 bursts with tomato-kombu umami against creamy burrata\u2014pure texture play and novelty. A great at-home experiment that aligns with your experimental/modernist dining streak."
        },
        {
          "name": "Tempura Maitake with Brown Butter Ponzu",
          "reasoning": "Ultra-light, airy batter delivers crisp contrast to earthy, frilly maitake, finished with nutty-brown butter and bright ponzu. It scratches your texture-driven, earthy/nutty flavor preference while staying focused and balanced."
        }
      ]
    }

### **Question 3**: Based on the workflow design, which phase takes the longest time to execute and why?


#### **Answer**: You can use this cell to enter your answer.



## Evaluate the recommendations


Create a simple evaluation function to assess recommendation quality.



```python
def evaluate_recommendations(result: Dict[str, Any]):
    """Evaluate the quality of recommendations."""
    print("\n" + "="*80)
    print("RECOMMENDATION EVALUATION")
    print("="*80)
    
    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    
    # Check if recommendations exist
    restaurants = recommendations.get("restaurants", [])
    recipes = recommendations.get("recipes", [])
    
    print(f"\n✓ Number of restaurant recommendations: {len(restaurants)}")
    print(f"✓ Number of recipe recommendations: {len(recipes)}")
    
    # Check dietary compliance
    dietary_restrictions = profile.get("dietary_restrictions", [])
    if dietary_restrictions:
        print(f"\n✓ Dietary restrictions identified: {', '.join(dietary_restrictions)}")
        print("  → Check if recommendations respect these restrictions")
    
    # Check diversity
    favorite_cuisines = profile.get("favorite_cuisines", [])
    if favorite_cuisines:
        print(f"\n✓ Favorite cuisines: {', '.join(favorite_cuisines)}")
        print("  → Check if recommendations include these cuisines")
    
    # Evaluate reasoning quality
    if restaurants:
        print(f"\n✓ First restaurant recommendation:")
        print(f"  Name: {restaurants[0].get('name', 'N/A')}")
        print(f"  Reasoning: {restaurants[0].get('reasoning', 'N/A')}")
    
    if recipes:
        print(f"\n✓ First recipe recommendation:")
        print(f"  Name: {recipes[0].get('name', 'N/A')}")
        print(f"  Reasoning: {recipes[0].get('reasoning', 'N/A')}")
    
    print("\n" + "="*80)
```


```python
# Evaluate Test Case 1 if available
try:
    if 'result_1' in locals():
        evaluate_recommendations(result_1)
except Exception as e:
    print(f"Evaluation requires completed test run: {e}")
```

    
    ================================================================================
    RECOMMENDATION EVALUATION
    ================================================================================
    
    ✓ Number of restaurant recommendations: 5
    ✓ Number of recipe recommendations: 5
    
    ✓ Dietary restrictions identified: Vegan, Gluten-Free
      → Check if recommendations respect these restrictions
    
    ✓ Favorite cuisines: Vegan/Plant-based, Mediterranean, Juice Bars/Smoothies
      → Check if recommendations include these cuisines
    
    ✓ First restaurant recommendation:
      Name: Zest & Zaatar
      Reasoning: All-vegan Mediterranean with a dedicated gluten-free falafel fryer hits your vegan/GF needs while keeping flavors bright with lemon, sumac, herbs, and garlicky tahini. It’s ideal for quick weekday lunches or casual dinners at a comfortable $$ price.
    
    ✓ First recipe recommendation:
      Name: Lemon-Tahini Quinoa Tabbouleh
      Reasoning: Quinoa keeps it naturally gluten-free while parsley, mint, and lemon–garlic tahini deliver the herbaceous, tangy profile you love. It’s a 20-minute, light lunch staple with clean, fresh flavors.
    
    ================================================================================

### **Question 4**: What is the purpose of the `workflow_step` field in the AgentState?


#### **Answer**: You can use this cell to enter your answer.



### Congratulations!
You have successfully completed this lab.


### Summary

In this lab, you implemented a multi-agent recommendation system with a hybrid workflow:
1. **Phase 1 (Sequential)**: User Profile Generator analyzes user data
2. **Phase 2 (Sequential)**: RAG Retriever queries the vector database
3. **Phase 3 (Parallel)**: Food Trend Analyst, Food Style Expert, and Nutrition Expert analyze candidates simultaneously
4. **Phase 4 (Sequential)**: Recommendation Expert synthesizes all insights into final recommendations

You used LangGraph to build a stateful workflow with shared state, node functions, and edges. You tested the system with multiple user personas and evaluated recommendation quality. In the next lesson, you will build a chatbot interface using Gradio to make the system interactive and user-friendly.


## Authors


[Tenzin Migmar](https://author.skills.network/instructors/tenzin_migmar)


©IBM Corporation. All rights reserved.


<!--## Change Log
|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2026-02-03|1|Tenzin Migmar|Create lab|
|2026-02-10|2|Jojy John|ID Reviewed|-->

