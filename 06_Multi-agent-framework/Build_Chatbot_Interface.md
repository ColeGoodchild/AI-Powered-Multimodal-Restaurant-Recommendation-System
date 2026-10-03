<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# Build a Chatbot Interface for the Recommendation System


Estimated time needed: **45** minutes


## Learning Objectives
After completing this lab, you'll be able to:

- Build interactive chatbot interfaces using Gradio
- Implement natural language intent classification for user requests
- Extract user preferences from conversational input
- Integrate multi-agent workflows into a chat interface
- Format agent outputs for display in a user-friendly format
- Add database management features to the chatbot
- Deploy and test a complete chatbot application


## Introduction


In this lab, you will build a chatbot interface for your multi-agent recommendation system using Gradio. The chatbot will:

1. Understand user requests in natural language
2. Classify whether users want restaurant or recipe recommendations
3. Extract preferences like cuisines, dietary restrictions, and price range
4. Invoke the multi-agent workflow from Lesson 2
5. Present recommendations in a conversational, engaging format
6. Allow users to add, update, and delete items in the database

By the end of this lab, you will have a fully functional chatbot that anyone can use without writing code.

**Note:** Throughout this lab, you'll answer strategically placed questions designed to reinforce your learning.


## Table of Contents

<font size = 3>    

1. [Install the required libraries](#Install-the-required-libraries)
2. [Import required libraries](#Import-required-libraries)
3. [Build a basic chatbot](#Build-a-basic-chatbot)
4. [Question 1: What two parameters does the chatbot function receive from Gradio's ChatInterface?](#Question-1:-What-two-parameters-does-the-chatbot-function-receive-from-Gradio's-ChatInterface?)
5. [Implement intent classification](#Implement-intent-classification)
6. [Extract user preferences](#Extract-user-preferences)
    1. [Task 1: Test the preference extraction function](#Task-1:-Test-the-preference-extraction-function)
7. [Integrate the multi-agent workflow](#Integrate-the-multi-agent-workflow)
8. [Format the recommendations](#Format-the-recommendations)
9. [Build a complete chatbot](#Build-a-complete-chatbot)
10. [Question 2: What are the five possible intent classifications the chatbot can identify?](#Question-2:-What-are-the-five-possible-intent-classifications-the-chatbot-can-identify?)
11. [Add database management](#Add-database-management)
12. [Deploy and test](#Deploy-and-test)
13. [Launch the chatbot](#Launch-the-chatbot)
14. [Question 3: What are the four tabs in the complete Gradio interface?](#Question-3:-What-are-the-four-tabs-in-the-complete-Gradio-interface?)
15. [Test the chatbot](#Test-the-chatbot)
     1. [Task 2: Test the chatbot with a recipe request](#Task-2:-Test-the-chatbot-with-a-recipe-request)


</font>
</div>


## **Screenshot requirement for this lab**

You will be prompted to take a screenshot and save it on your own device. You will need this screenshot either to answer graded quiz questions or to upload as your submission for the Final Project at the end of this course. You can use various free screen-grabbing tools or your operating system's shortcut keys to do this (for example, `Alt+PrintScreen` on Windows and `Command+shift+4` on Mac).
**Note**: The screenshot can be saved with either the **.jpg** or **.png** extension.


### Install the required libraries

All the required libraries are __not__ pre-installed in the Skills Network Labs environment. __You need to run the following cell__ to install them, and this might take a few minutes.
**Restart the Kernel** once the install completed.


----



```python
%%capture
!pip install --upgrade pip

# Core UI (stable with Python 3.12)
!pip install gradio==4.29.0
!pip install jinja2==3.1.4

# LangChain stack (compatible set)
!pip install langchain==0.2.16
!pip install langchain-openai==0.1.7
!pip install langchain-core==0.2.38

!pip install langgraph==0.1.19
!pip install huggingface_hub==0.23.5


!pip install fastapi==0.110.0
!pip install starlette==0.36.3
```

### Import required libraries



```python
import gradio as gr
import json
from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
```


```python
# Configure OpenAI API
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
```

## Build a basic chatbot

Start with a simple chatbot that echoes user messages. This will help you understand Gradio's chat components.

**Note**: You may see a warning notification indicating that a newer version of Gradio is available. You can safely ignore this warning and proceed with the lab.



```python
from typing import List, Tuple
import gradio as gr

def echo_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    """A simple echo chatbot for testing."""
    return f"You said: {message}"

# Create interface
demo_echo = gr.ChatInterface(
    fn=echo_chatbot,
    title="Echo Chatbot",
    description="A simple chatbot that echoes your messages.",
    examples=[
        "Hello!",
        "How are you?",
        "Tell me about restaurants."
    ]
)

print("Basic chatbot interface created!")
print("To launch, run: demo_echo.launch()")
```

    Basic chatbot interface created!
    To launch, run: demo_echo.launch()

### **Question 1**: What two parameters does the chatbot function receive from Gradio's ChatInterface?


#### **Answer**: You can use this cell to enter your answer.


## Implement intent classification

Now, add logic to classify user intent: Do they want restaurant recommendations, recipe recommendations, or both?



```python
def classify_intent(user_message: str, llm: ChatOpenAI) -> str:
    """Classify user intent as restaurant, recipe, both, or clarification."""
    
    system_prompt = """You are an intent classifier for a food recommendation system.
    
Analyze the user's message and classify it as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe" - User wants recipe recommendations
- "both" - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database" - User wants to add/edit/delete database entries

Examples:
"Where should I eat tonight?" → restaurant
"How do I make lasagna?" → recipe
"I want dinner ideas" → both
"What can you help me with?" → clarification
"I want to add a new restaurant" → database

Respond with ONLY the classification label."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    intent = response.content.strip().lower()
    
    # Validate intent
    valid_intents = ["restaurant", "recipe", "both", "clarification", "database"]
    if intent not in valid_intents:
        intent = "clarification"
    
    return intent

print("Intent classification function created!")
```

    Intent classification function created!


```python
# Test intent classification
test_messages = [
    "I'm looking for Italian restaurants",
    "How do I make pad thai?",
    "Give me dinner ideas",
    "What can you do?"
]

print("Testing intent classification:\n")
try:
    for msg in test_messages:
        intent = classify_intent(msg, llm)
        print(f"Message: '{msg}'")
        print(f"Intent: {intent}\n")
except Exception as e:
    print(f"Intent classification requires valid OpenAI API key. Error: {e}")
```

    Testing intent classification:
    Message: 'I'm looking for Italian restaurants'
    Intent: restaurant
    Message: 'How do I make pad thai?'
    Intent: recipe
    Message: 'Give me dinner ideas'
    Intent: both
    Message: 'What can you do?'
    Intent: clarification


## Extract user preferences

Once you know the user's intent, you need to extract their preferences from natural language.



```python
def extract_preferences(user_message: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Extract user preferences from natural language input."""
    
    system_prompt = """You are a preference extractor for a food recommendation system.
    
Extract user preferences from their message and return JSON with these keys:
- favorite_cuisines: List of mentioned cuisines (e.g., ["Italian", "Thai"])
- dietary_restrictions: List of dietary needs (e.g., ["vegetarian", "gluten-free"])
- dining_occasion: Type of dining (e.g., "casual", "fine dining", "quick bite")
- price_range: Price preference (e.g., "$", "$$", "$$$", "$$$$")
- flavor_preferences: List of flavor preferences (e.g., ["spicy", "sweet"])
- other_preferences: Any other relevant details

If a field is not mentioned, use an empty list or "not specified".

Example:
Input: "I love spicy Thai food and I'm vegetarian"
Output: {
  "favorite_cuisines": ["Thai"],
  "dietary_restrictions": ["vegetarian"],
  "dining_occasion": "not specified",
  "price_range": "not specified",
  "flavor_preferences": ["spicy"],
  "other_preferences": ""
}

Respond with ONLY valid JSON."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    try:
        preferences = json.loads(response.content)
    except:
        # Fallback if parsing fails
        preferences = {
            "favorite_cuisines": [],
            "dietary_restrictions": [],
            "dining_occasion": "not specified",
            "price_range": "not specified",
            "flavor_preferences": [],
            "other_preferences": ""
        }
    
    return preferences

print("Preference extraction function created!")
```

    Preference extraction function created!

### **Task 1**: Test the preference extraction function

Test the `extract_preferences` function with the message: "I'm looking for affordable vegetarian Mexican restaurants"



```python
## Type your answer here

test_message = "I'm looking for affordable vegetarian Mexican restaurants"
try:
    preferences = extract_preferences(test_message, llm)
    print("Extracted Preferences:")
    print(json.dumps(preferences, indent=2))
except Exception as e:
    print(f"Requires valid OpenAI API key. Error: {e}")
```

    Extracted Preferences:
    {
      "favorite_cuisines": [
        "Mexican"
      ],
      "dietary_restrictions": [
        "vegetarian"
      ],
      "dining_occasion": "not specified",
      "price_range": "$",
      "flavor_preferences": [],
      "other_preferences": ""
    }

Take a screenshot of the Python code, clearly showing your implementation from Task 1. Name the screenshot ```M3L3_preference_extraction_test.jpg```.


## Integrate the multi-agent workflow

Now we'll create a simplified version of the multi-agent workflow that can be called from the chatbot.



```python
def run_recommendation_workflow(preferences: Dict[str, Any], recommendation_type: str) -> Dict[str, Any]:
    """Run the multi-agent workflow and return recommendations.
    
    Args:
        preferences: User preferences extracted from their message
        recommendation_type: "restaurant", "recipe", or "both"
    
    Returns:
        Dictionary with recommendations
    """
    
    # For this demo, we'll create mock recommendations
    # In a real implementation, this would call the LangGraph workflow from Lesson 2
    
    print(f"Running workflow for {recommendation_type} recommendations...")
    
    mock_recommendations = {
        "restaurants": [
            {
                "name": "Green Leaf Bistro",
                "cuisine": "Mediterranean",
                "price": "$$",
                "reasoning": "This restaurant perfectly aligns with your preference for healthy, plant-based options and offers a diverse Mediterranean menu with excellent vegetarian choices."
            },
            {
                "name": "Spice Route",
                "cuisine": "Indian",
                "price": "$$",
                "reasoning": "Known for authentic Indian cuisine with extensive vegetarian options. The spice level can be customized to your preference."
            }
        ],
        "recipes": [
            {
                "name": "One-Pot Chickpea Curry",
                "cuisine": "Indian",
                "difficulty": "Easy",
                "reasoning": "A flavorful, protein-rich dish that matches your love for bold flavors. Ready in 30 minutes with simple ingredients."
            },
            {
                "name": "Mediterranean Quinoa Bowl",
                "cuisine": "Mediterranean",
                "difficulty": "Easy",
                "reasoning": "Nutritious and satisfying, this bowl combines your favorite Mediterranean flavors with plant-based protein."
            }
        ]
    }
    
    # Filter based on recommendation type
    if recommendation_type == "restaurant":
        return {"restaurants": mock_recommendations["restaurants"]}
    elif recommendation_type == "recipe":
        return {"recipes": mock_recommendations["recipes"]}
    else:  # both
        return mock_recommendations

print("Workflow integration function created!")
```

    Workflow integration function created!

## Format the recommendations

Create a function to format recommendations in a user-friendly way.



```python
def format_recommendations(recommendations: Dict[str, Any]) -> str:
    """Format recommendations for display in the chat."""
    
    output = ""
    
    # Format restaurant recommendations
    if "restaurants" in recommendations and recommendations["restaurants"]:
        output += "🍽️ **Restaurant Recommendations:**\n\n"
        for i, restaurant in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {restaurant['name']}**\n"
            output += f"   - Cuisine: {restaurant['cuisine']}\n"
            output += f"   - Price: {restaurant['price']}\n"
            output += f"   - Why: {restaurant['reasoning']}\n\n"
    
    # Format recipe recommendations
    if "recipes" in recommendations and recommendations["recipes"]:
        output += "👨‍🍳 **Recipe Recommendations:**\n\n"
        for i, recipe in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {recipe['name']}**\n"
            output += f"   - Cuisine: {recipe['cuisine']}\n"
            output += f"   - Difficulty: {recipe['difficulty']}\n"
            output += f"   - Why: {recipe['reasoning']}\n\n"
    
    if not output:
        output = "I couldn't generate recommendations. Please try again with more details about your preferences."
    
    return output

print("Formatting function created!")
```

    Formatting function created!

## Build a complete chatbot

Now, combine everything into a complete chatbot.



```python
def recommendation_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    """Main chatbot function that handles user requests."""
    
    try:
        # Step 1: Classify intent
        intent = classify_intent(message, llm)
        print(f"Classified intent: {intent}")
        
        # Step 2: Handle different intents
        if intent == "clarification":
            return """I'm your food recommendation assistant! I can help you with:
            
🍽️ **Restaurant recommendations** - Tell me your cuisine preferences, dietary restrictions, and occasion
👨‍🍳 **Recipe recommendations** - Let me know what you'd like to cook
📝 **Database management** - Add, update, or delete restaurants and recipes

Just describe what you're looking for, and I'll provide personalized recommendations!"""
        
        elif intent == "database":
            return """To manage the database, please use the tabs above:
            
- **Add Restaurant**: Submit a new restaurant
- **Add Recipe**: Submit a new recipe
- **Edit/Delete**: Modify or remove existing entries

Is there anything else I can help you with?"""
        
        elif intent in ["restaurant", "recipe", "both"]:
            # Step 3: Extract preferences
            preferences = extract_preferences(message, llm)
            print(f"Extracted preferences: {preferences}")
            
            # Step 4: Run workflow
            recommendations = run_recommendation_workflow(preferences, intent)
            
            # Step 5: Format output
            formatted_output = format_recommendations(recommendations)
            
            return formatted_output
        
        else:
            return "I'm not sure how to help with that. Can you rephrase your request?"
    
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please make sure you have set your OpenAI API key."

print("Complete chatbot function created!")
```

    Complete chatbot function created!

### **Question 2**: What are the five possible intent classifications the chatbot can identify?


#### **Answer**: You can use this cell to enter your answer.



## Add database management

Next, add functions for database management (add, update, and delete).



```python
def add_restaurant(name: str, cuisine: str, price: str, location: str, description: str) -> str:
    """Add a new restaurant to the database."""
    # In a real implementation, this would add to the vector database
    print(f"Adding restaurant: {name}")
    return f"✅ Successfully added '{name}' to the database!"

def add_recipe(name: str, cuisine: str, difficulty: str, prep_time: str, ingredients: str, instructions: str) -> str:
    """Add a new recipe to the database."""
    # In a real implementation, this would add to the vector database
    print(f"Adding recipe: {name}")
    return f"✅ Successfully added '{name}' recipe to the database!"

print("Database management functions created!")
```

    Database management functions created!

## Deploy and test

Next, create the complete Gradio interface with multiple tabs.



```python
# Create the main interface with tabs
with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🍽️ Food Recommendation Chatbot
    
    Your personal AI assistant for restaurant and recipe recommendations!
    """)
    
    with gr.Tabs():
        
        # Tab 1: Chat Interface
        with gr.Tab("💬 Chat"):
            chatbot_interface = gr.ChatInterface(
                fn=recommendation_chatbot,
                examples=[
                    "I'm looking for vegetarian restaurants",
                    "Suggest some easy recipes for dinner",
                    "I want spicy Thai food recommendations",
                    "What can you help me with?"
                ],
                title="Chat with the Recommendation Assistant",
                description="Describe your food preferences and I'll recommend restaurants or recipes!"
            )
        
        # Tab 2: Add Restaurant
        with gr.Tab("➕ Add Restaurant"):
            gr.Markdown("### Add a New Restaurant to the Database")
            
            with gr.Row():
                with gr.Column():
                    rest_name = gr.Textbox(label="Restaurant Name")
                    rest_cuisine = gr.Textbox(label="Cuisine Type")
                    rest_price = gr.Dropdown(
                        choices=["$", "$$", "$$$", "$$$$"],
                        label="Price Range"
                    )
                with gr.Column():
                    rest_location = gr.Textbox(label="Location")
                    rest_description = gr.Textbox(
                        label="Description",
                        lines=3
                    )
            
            add_rest_btn = gr.Button("Add Restaurant", variant="primary")
            rest_output = gr.Textbox(label="Status")
            
            add_rest_btn.click(
                fn=add_restaurant,
                inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_description],
                outputs=rest_output
            )
        
        # Tab 3: Add Recipe
        with gr.Tab("➕ Add Recipe"):
            gr.Markdown("### Add a New Recipe to the Database")
            
            with gr.Row():
                with gr.Column():
                    recipe_name = gr.Textbox(label="Recipe Name")
                    recipe_cuisine = gr.Textbox(label="Cuisine Type")
                    recipe_difficulty = gr.Dropdown(
                        choices=["Easy", "Medium", "Hard"],
                        label="Difficulty"
                    )
                with gr.Column():
                    recipe_time = gr.Textbox(label="Prep Time")
                    recipe_ingredients = gr.Textbox(
                        label="Ingredients (comma-separated)",
                        lines=3
                    )
            
            recipe_instructions = gr.Textbox(
                label="Instructions",
                lines=5
            )
            
            add_recipe_btn = gr.Button("Add Recipe", variant="primary")
            recipe_output = gr.Textbox(label="Status")
            
            add_recipe_btn.click(
                fn=add_recipe,
                inputs=[recipe_name, recipe_cuisine, recipe_difficulty, recipe_time, recipe_ingredients, recipe_instructions],
                outputs=recipe_output
            )
        
        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Chatbot
            
            This chatbot uses a multi-agent AI system to provide personalized food recommendations.
            
            ### Features:
            - 🤖 **Intelligent Agents**: Six specialized AI agents work together to analyze your preferences
            - 🔍 **Smart Search**: Vector database retrieval finds the most relevant options
            - 🎯 **Personalized**: Recommendations tailored to your tastes and dietary needs
            - 📝 **Editable Database**: Add your favorite restaurants and recipes
            
            ### How to Use:
            1. Go to the **Chat** tab
            2. Describe what you're looking for (cuisine, dietary restrictions, occasion, etc.)
            3. Receive personalized restaurant or recipe recommendations
            4. Use the **Add** tabs to contribute to the database
            
            ### Technologies:
            - LangChain & LangGraph for multi-agent orchestration
            - OpenAI GPT-4 for language understanding
            - Vector databases for semantic search
            - Gradio for the user interface
            """)

print("Complete interface created!")
print("\nTo launch the chatbot, run: demo.launch()")
```

    Complete interface created!
    
    To launch the chatbot, run: demo.launch()

### Launch the chatbot



```python
# Uncomment the line below to launch the chatbot
demo.launch(share=True)

print("To launch the chatbot interface, uncomment and run the line above.")
print("The chatbot will open in a new browser tab.")
```

    Running on local URL:  http://127.0.0.1:7860Running on public URL: https://a0791fa89c37ca8489.gradio.live
    
    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)


<div><iframe src="https://a0791fa89c37ca8489.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


    To launch the chatbot interface, uncomment and run the line above.
    The chatbot will open in a new browser tab.

### **Question 3**: What are the four tabs in the complete Gradio interface?


#### **Answer**: You can use this cell to enter your answer.



## Test the chatbot

Test the chatbot logic with a sample interaction.



```python
# Test the chatbot with a sample message
test_message = "I'm looking for healthy vegetarian restaurants for a date night"

print("Testing chatbot with message:")
print(f"User: {test_message}\n")

try:
    response = recommendation_chatbot(test_message, [])
    print("Bot Response:")
    print(response)
except Exception as e:
    print(f"Test requires valid OpenAI API key. Error: {e}")
```

    Testing chatbot with message:
    User: I'm looking for healthy vegetarian restaurants for a date night
    Classified intent: restaurantExtracted preferences: {'favorite_cuisines': ['vegetarian'], 'dietary_restrictions': ['vegetarian'], 'dining_occasion': 'date night', 'price_range': 'not specified', 'flavor_preferences': ['healthy'], 'other_preferences': ''}
    Running workflow for restaurant recommendations...
    Bot Response:
    🍽️ **Restaurant Recommendations:**
    
    **1. Green Leaf Bistro**
       - Cuisine: Mediterranean
       - Price: $$
       - Why: This restaurant perfectly aligns with your preference for healthy, plant-based options and offers a diverse Mediterranean menu with excellent vegetarian choices.
    
    **2. Spice Route**
       - Cuisine: Indian
       - Price: $$
       - Why: Known for authentic Indian cuisine with extensive vegetarian options. The spice level can be customized to your preference.
    


### **Task 2**: Test the chatbot with a recipe request

Create a test message asking for recipe recommendations and run it through the chatbot.



```python
## Type your answer here

test_recipe_message = "Can you give me some recipe recommendations?"

print("Testing chatbot with recipe request:")
print(f"User: {test_recipe_message}\n")

try:
    response = recommendation_chatbot(test_recipe_message, [])
    print("Bot Response:")
    print(response)
except Exception as e:
    print(f"Test requires valid OpenAI API key. Error: {e}")
```

    Testing chatbot with recipe request:
    User: Can you give me some recipe recommendations?
    Classified intent: recipeExtracted preferences: {'favorite_cuisines': [], 'dietary_restrictions': [], 'dining_occasion': 'not specified', 'price_range': 'not specified', 'flavor_preferences': [], 'other_preferences': ''}
    Running workflow for recipe recommendations...
    Bot Response:
    👨‍🍳 **Recipe Recommendations:**
    
    **1. One-Pot Chickpea Curry**
       - Cuisine: Indian
       - Difficulty: Easy
       - Why: A flavorful, protein-rich dish that matches your love for bold flavors. Ready in 30 minutes with simple ingredients.
    
    **2. Mediterranean Quinoa Bowl**
       - Cuisine: Mediterranean
       - Difficulty: Easy
       - Why: Nutritious and satisfying, this bowl combines your favorite Mediterranean flavors with plant-based protein.
    


### Congratulations!
You have successfully completed this lab.


### Summary

In this lab, you built a complete chatbot interface for your multi-agent recommendation system using Gradio. You:

1. Created a basic chat interface to understand Gradio components
2. Implemented intent classification to understand user requests
3. Built preference extraction to convert natural language to structured data
4. Integrated the multi-agent workflow from Lesson 2
5. Formatted recommendations for user-friendly display
6. Added database management features for adding restaurants and recipes
7. Created a multi-tab interface with chat, add forms, and information sections
8. Tested the chatbot with various user inputs

You now have a fully functional, production-ready chatbot that makes your AI system accessible to anyone, regardless of technical expertise. This completes your multi-agent recommendation system!


### Thank you for completing this lab!



## Authors 

[Tenzin Migmar](https://author.skills.network/instructors/tenzin_migmar)


©IBM Corporation. All rights reserved.


<!--## Change Log
|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2026-02-03|1|Tenzin Migmar|Create lab|
|2026-02-10|2|Jojy John|ID Reviewed|-->

