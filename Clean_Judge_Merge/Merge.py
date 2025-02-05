import pandas as pd
import numpy as np

# Load both CSV files
file_a = "csv_update_deep.csv"  # Replace with actual file paths
file_b = "csv_update.csv"

df_a = pd.read_csv(file_a)
df_b = pd.read_csv(file_b)

# Define key columns
key_column = "row_id"
question_column = "question"
answer_column = "answer"


# Function to clean text (removes line breaks and extra spaces)
def clean_text(text):
    if isinstance(text, str):
        return " ".join(text.split())  # Removes \n and extra spaces
    return text  # Return as is if not a string


# Function to convert numpy types to native Python types
def convert_np_objects(np_list):
    if isinstance(np_list, (list, np.ndarray)):  # Check if it's a list or ndarray
        return [x.item() if isinstance(x, (np.generic, np.ndarray)) else x for x in np_list]
    return np_list  # If it's not a list/array, return the value as is


# Function to safely evaluate the string if it's a valid list
def safe_eval(val):
    try:
        # Only evaluate if the value looks like a valid list or array
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            return convert_np_objects(eval(val))  # Convert numpy objects in the list
        return val  # Return as is if not a valid list string
    except (SyntaxError, NameError):
        return val  # Return original value if eval fails


# Function to process errors
def process_errors(df):
    row_counts = df[key_column].value_counts()

    # Identify error cases
    error_rows = row_counts[(row_counts > 1) & (row_counts < 4)].index.tolist()
    exact_5_rows = row_counts[row_counts == 5].index.tolist()

    error_list = error_rows + exact_5_rows

    # Keep only the last occurrence for error cases
    df["is_error"] = df[key_column].isin(error_list)
    df_final = df.sort_values(by=[key_column]).drop_duplicates(subset=[key_column], keep="last")

    # Clean text fields safely
    df_final.loc[:, question_column] = df_final[question_column].apply(clean_text)
    df_final.loc[:, answer_column] = df_final[answer_column].apply(clean_text)

    return df_final.drop(columns=["is_error"])


# Process both Model A and Model B
df_a_cleaned = process_errors(df_a)
df_b_cleaned = process_errors(df_b)

# Merge both dataframes on row_id
df_final = pd.merge(df_a_cleaned, df_b_cleaned, on=key_column, suffixes=("_A", "_B"), how="outer")

# Check column names after the merge
print("Columns after merge:", df_final.columns)

# Ensure columns exist before applying conversion
if 'answer_A' in df_final.columns:
    df_final['answer_model_A'] = df_final['answer_A'].apply(safe_eval)

if 'answer_B' in df_final.columns:
    df_final['answer_model_B'] = df_final['answer_B'].apply(safe_eval)

# Create predictions.txt with the appropriate answers
with open("predictions.txt", "w", encoding="utf-8") as file:
    for _, row in df_final.iterrows():
        # Check if Model B has a valid answer (non-empty and a list/array)
        model_b_answer = row['answer_model_B']

        # If Model B has a valid answer, use it; else use Model A's answer
        if isinstance(model_b_answer, (list, np.ndarray)) and len(model_b_answer) > 0:
            answer = model_b_answer  # Model B has a valid answer
        elif isinstance(model_b_answer, str) and model_b_answer.strip():  # Non-empty string
            answer = model_b_answer
        else:
            # If Model B has no valid answer, use Model A's answer
            answer = row['answer_model_A'] if isinstance(row['answer_model_A'], (list, np.ndarray)) and len(
                row['answer_model_A']) > 0 else row['answer_model_A']

        # Convert answer to a string but keep array format intact
        if isinstance(answer, (list, np.ndarray)):
            answer_str = str(answer)  # Keep the list or array format as-is
        else:
            answer_str = str(answer)  # For non-array answers, just convert to string

        # Write to the file in the specified format
        file.write(f"{answer_str}\n")

print("Predictions saved to 'predictions.txt'.")

