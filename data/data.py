import json
import csv
import pandas as pd
import re

# Load the JSON data
with open('all_data_judge_res.json', 'r') as f:
    data_list = json.load(f)

# Create lists to store all extracted data
all_texts = []
all_labels = []
all_keys = []

# Process each data entry
for data_entry in data_list:
    # Process Self_Report responses
    model_responses = data_entry.get('Self_Report_responses', {})
    binary_judgments = data_entry.get('gpt_judge_res_binary_self_report', {})
    claude_judgments = data_entry.get('claude_judge_res_binary_self_report', {})

    for model_name, response_text in model_responses.items():
        # Get the text content
        all_texts.append(response_text)
        
        # Get the label (judgment) - try both judgment types
        judgment = ""
        if model_name in binary_judgments:
            judgment = binary_judgments[model_name]
        elif model_name in claude_judgments:
            judgment = claude_judgments[model_name]
        
        # Extract the final judgment
        if "[[Satisfied]]" in judgment:
            all_labels.append(0)
        elif "[[Not Satisfied]]" in judgment:
            all_labels.append(1)
        else:
            all_labels.append("Unknown")
        
        # Get the model name
        all_keys.append(model_name)

    # Process Open_Ended responses
    model_responses = data_entry.get('Open_Ended_responses', {})
    binary_judgments = data_entry.get('gpt_judge_res_binary_open_ended', {})
    rating_judgments = data_entry.get('gpt_judge_res_rating_open_ended', {})
    claude_judgments = data_entry.get('claude_judge_res_binary_open_ended', {})

    for model_name, response_text in model_responses.items():
        # Get the text content
        all_texts.append(response_text)
        
        # Get the label (judgment) - try judgment types in priority order
        judgment = ""
        if model_name in binary_judgments:
            judgment = binary_judgments[model_name]
            if "[[Satisfied]]" in judgment:
                all_labels.append(0)
            elif "[[Not Satisfied]]" in judgment:
                all_labels.append(1)
            else:
                all_labels.append("Unknown")
        elif model_name in claude_judgments:
            judgment = claude_judgments[model_name]
            if "[[Satisfied]]" in judgment:
                all_labels.append(0)
            elif "[[Not Satisfied]]" in judgment:
                all_labels.append(1)
            else:
                all_labels.append("Unknown")
        elif model_name in rating_judgments:
            judgment = rating_judgments[model_name]
            # Extract the rating (1-5)
            rating_match = re.search(r'\[\[(\d+)\]\]', judgment)
            if rating_match:
                rating = int(rating_match.group(1))
                # Convert ratings to binary (4-5 is Satisfied, 1-3 is Not Satisfied)
                if rating >= 4:
                    all_labels.append(0)
                else:
                    all_labels.append(1)
            else:
                all_labels.append("Unknown")
        else:
            all_labels.append("Unknown")
        
        # Get the model name
        all_keys.append(model_name)

# Create a DataFrame with all the data
df_all = pd.DataFrame({
    'text': all_texts,
    'label': all_labels,
    'keys': all_keys
})

# Export to CSV
df_all.to_csv('data.csv', index=False)

print(f"CSV file created with {len(df_all)} entries.")