from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $'
EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}"""

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = """
    You are a helpful AI assistant that specializes in transforming raw input data into a specific JSON format. Do not deviate from this format. Do not make up variables. If a variable cannot be inferred from the input data, leave it blank instead of guessing. Output only the JSON.
    """

    base_user_prompt = f"""
    Task:
    Transform raw input data into the required structured JSON format. Do not deviate from this template. Do not make up variables. Convert dollar signs (e.g., $, $$) into an integer representing the number of dollar symbols.

    Restaurant description: {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}

    Output:
    {EXAMPLE_OUTPUT}
    """
    return base_system_msg, base_user_prompt

# Granite is used here because it is a small, cheap model that is good enough for simple structuring tasks
def llm_model(system_msg, prompt_txt, params=None):
    model_id = "ibm/granite-4-h-small"
    project_id = "skills-network"
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    response = model.chat(messages=messages, params=params)
    return response["choices"][0]["message"]["content"]

def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """
    You are a JSON repair expert. Your only job is to fix malformed or schema-invalid JSON from the input so that it passes validation.

    Rules:
    - Output ONLY the corrected JSON. No explanations, no commentary, no markdown code fences
    - Preserve the original data and meaning. Do not invent, remove, or rename fields unless the error message requires it
    - Fix syntax problems (missing quotes, trailing commas, unbalanced brackets) and schema problems as indicated
    - If a required text value is missing and cannot be inferred from the input, use an empty string; for optional fields, use null
    """
    auto_repair_prompt = f"""
    Original (incorrect) output:
    {response}

    Validation error message:
    {error_message}

    Use the error message as guidance to correct the output. Return only the corrected, valid JSON.
    """
    return auto_repair_system_msg, auto_repair_prompt

def new_data_entry_process(paragraph, itemId, max_retries=3):
    # 1. Generate the initial structured output
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    candidate_json_output = llm_model(system_msg, user_prompt)

    # 2. Validate, and auto-repair on failure (capped to prevent infinite loops)
    for attempt in range(max_retries + 1):
        # Strip markdown code fences if the model added them
        candidate_json_output = candidate_json_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            restaurant = Restaurant.model_validate_json(candidate_json_output)
            break
        except ValidationError as e:
            if attempt == max_retries:
                raise ValueError(f"Could not produce valid JSON after {max_retries} repairs: {e}")
            repair_sys, repair_prompt = JSON_auto_repair_prompts(candidate_json_output, e.json())
            candidate_json_output = llm_model(repair_sys, repair_prompt)

    # 3. Add the item ID and return the finished record
    result = restaurant.model_dump()
    result['itemId'] = itemId
    return result

# --- Helper functions (replace with the course's versions if it provided them) ---
def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, file_path, backup_path=None):
    if backup_path and os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def show_restaurant_card(res, index):
    print(f"\n--- Restaurant #{index} ---")
    for key, value in res.items():
        print(f"{key}: {value}")

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                print(f"{i}. {res.get('name', 'N/A')}")

        elif choice == '2':
            index_input = str(input("Enter record index: "))
            if index_input.isdigit() and 0 <= int(index_input) < len(data):
                index = int(index_input)
                show_restaurant_card(data[index], index)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1  # the item id for the new data

                paragraph = input("Enter a new restaurant description: ")
                new_restaurant = new_data_entry_process(paragraph, itemId)
                data.append(new_restaurant)
                save_data(data, file_path, backup_path)

                print("✅ Restaurant added.")

            elif choice == '4':  # EDIT DATA
                index_input = str(input("Enter record index to edit: "))
                if index_input.isdigit() and 0 <= int(index_input) < len(data):
                    record = data[int(index_input)]
                    for key in record.keys():
                        new_value = input(f"{key} [{record[key]}] (Enter to skip): ")
                        if new_value == "":
                            continue
                        try:
                            if isinstance(record[key], int):
                                record[key] = int(new_value)
                            elif isinstance(record[key], float):
                                record[key] = float(new_value)
                            elif isinstance(record[key], list):
                                record[key] = [item.strip() for item in new_value.split(',')]
                            else:
                                record[key] = new_value
                        except ValueError:
                            print(f"Invalid value for {key}, keeping the old one.")
                    save_data(data, file_path, backup_path)
                    print("✅ Record updated.")
                else:
                    print("invalid index.")

            elif choice == '5':  # DELETE DATA
                index_input = str(input("Enter record index to delete: "))
                if index_input.isdigit() and 0 <= int(index_input) < len(data):
                    removed = data.pop(int(index_input))
                    save_data(data, file_path, backup_path)
                    print(f"✅ Deleted: {removed.get('name', 'N/A')}")
                else:
                    print("invalid index.")

        elif choice == '6':  # EXIT
            break
        else:
            print("Invalid input.")


class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant, then delete one.
        """
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)  # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())

if __name__ == "__main__":
    unittest.main()  # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH)  # Actual UI Call
