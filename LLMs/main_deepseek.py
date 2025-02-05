import os
import pandas as pd
import pyarrow.parquet as pq
import zipfile
import io
import sys
import time
from together import Together
from together.error import RateLimitError

# Initialize Together API client
client = Together(#)  # Replace with your API key
model_name = "deepseek-ai/DeepSeek-V3"


def clean_code(code):
    """Removes markdown-style Python code blocks."""
    return code.replace("python", "").replace("```", "").strip()


def get_llm_completion(prompt):
    """Send prompt to Together API and return the generated code."""
    try:
        print("Sending prompt to API...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.7,
            stream=False
        )
        result = response.choices[0].message.content.strip()
        result = clean_code(result)  # Remove markdown formatting
        print(f"Received API response:\n{result}\n")
        return result
    except RateLimitError as e:
        print(f"Rate limit hit: {e}. Retrying after 60 seconds...")
        time.sleep(60)
        return get_llm_completion(prompt)
    except Exception as e:
        print(f"Error: {e}")
        return ""


def load_dataset(dataset_name, base_path="competition"):
    """Loads the dataset as a Pandas DataFrame from Parquet files."""
    dataset_path = os.path.join(base_path, dataset_name)
    file_path = os.path.join(dataset_path, "all.parquet")

    if os.path.exists(file_path):
        return pq.read_table(file_path).to_pandas()
    else:
        raise FileNotFoundError(f"No parquet file found: {file_path}")


def generate_prompt(question, dataset_name, df, previous_code=None, previous_error=None):
    """Generates a prompt for the LLM with dataset context and previous errors if applicable."""

    # Display a sample of the first few rows in the dataset for context
    sample_rows = df.head(3).to_dict(orient='records')

    prompt = f"""
        The following table is from the dataset {dataset_name}. The first few rows are:
        {sample_rows}

        Your task is to write Python code that answers the question: "{question}"
        The code should:
        - Load the dataset from the appropriate location. The dataset is in the folder named 'competition' which contains subfolders named after the datasets (e.g., {dataset_name}).
        - Load the dataset file (`all.parquet`), and perform the necessary operations on the DataFrame.

        Please do the following:
        - Use `pd.read_parquet` to load the dataset from the full data file (`all.parquet`).
        - Process the DataFrame named 'df' accordingly and print the final result.

        Please **return only the Python code**, without any explanation or extra text, and make sure it selects the correct file using the `dataset_name` from the 'competition' folder.

        The file path for the full dataset is:
        `competition/{dataset_name}/all.parquet`

        Do not include explanations, comments, or code block markers (e.g., ```python). If it has more than one answer, write the output like this example: ['United Kingdom', 'Germany', 'France']. Each code print must be in only one line (no enter) . If question answer was binary (True, False), Do not write the count of that in answer. Types of Answers Expected

    According to the expected answer types:

        Boolean: Valid answers include True/False, Y/N, Yes/No (all case insensitive).
        Category: A value from a cell (or a substring of a cell) in the dataset.
        Number: A numerical value from a cell in the dataset, which may represent a computed statistic (e.g., average, maximum, minimum).
        List[category]: A list containing a fixed number of categories. The expected format is: "['cat', 'dog']". Pay attention to the wording of the question to determine if uniqueness is required or if repeated values are allowed.
        List[number]: Similar to List[category], but with numbers as its elements.
    Also import all needed packages.
    You will not know the specific type of answer expected, but you can be assured that it will be one of these types. For the competition, the order of the elements within the list answers will not be taken into account. print output in the code must be one of the above answer types.
        """

    if previous_code and previous_error:
        prompt += f"""

        The previous attempt resulted in an error:
        {previous_error}

        The previous code was:
        {previous_code}

        Please correct the code and return only the fixed Python code.
        """
    return prompt


def update_csv(row_id, output, error_message, question, dataset_name):
    """Updates the csv_update_final.csv file with the latest output/error."""
    csv_update_path = "csv_update_deep.csv"

    # Read the existing CSV or create a new one
    if os.path.exists(csv_update_path):
        csv_df = pd.read_csv(csv_update_path)
    else:
        csv_df = pd.DataFrame(columns=["row_id", "question", "dataset", "answer", "error_message"])

    # Prepare a new row as a DataFrame
    new_row = pd.DataFrame([{
        "row_id": row_id,
        "question": question,
        "dataset": dataset_name,
        "answer": output,
        "error_message": error_message
    }])

    # Concatenate the new row with the existing DataFrame
    csv_df = pd.concat([csv_df, new_row], ignore_index=True)

    # Save the updated CSV
    csv_df.to_csv(csv_update_path, index=False)


def execute_code(prompt, df, question, dataset_name, row_id):
    """Executes generated Python code and captures the printed output."""
    max_retries = 5  # Max retry limit
    attempt = 0  # Initialize attempt counter

    while attempt < max_retries:  # Retry mechanism for error handling
        try:
            code = get_llm_completion(prompt)
            with open("generated_code_deep.py", "w") as f:
                f.write(code)

            output_capture = io.StringIO()
            sys.stdout = output_capture

            exec_locals = {"df": df}
            exec(code, {}, exec_locals)

            output = output_capture.getvalue().strip()
            sys.stdout = sys.__stdout__

            print(f"Code execution result: {output}\n")

            # Update the CSV file with the result
            #update_csv(row_id, output, "", question, dataset_name)

            return output  # Stop retrying if successful

        except Exception as e:
            sys.stdout = sys.__stdout__
            error_message = str(e)
            print(f"Execution Error: {error_message}")

            # Update the CSV file with the error
            update_csv(row_id, "", error_message, question, dataset_name)

            attempt += 1  # Increment the attempt counter

            if attempt == max_retries:  # If retries have been exhausted
                break  # Break out of the retry loop, continue with the next question

            # Regenerate prompt with error details for retry
            prompt = generate_prompt("", "", df, code, error_message)

    return None  # Return None if all retries failed (the loop has been exhausted)

def process_question(question, dataset_name, row_id):
    """Handles dataset loading, prompt generation, and error handling."""
    try:
        df = load_dataset(dataset_name)
        prompt = generate_prompt(question, dataset_name, df)
        return execute_code(prompt, df, question, dataset_name, row_id)
    except Exception as e:
        print(f"Error: {str(e)}")
        update_csv(row_id, "", str(e), question, dataset_name)
        return None


if os.path.exists("csv_update_deep.csv"):
    csv_update_df = pd.read_csv("csv_update_deep.csv")
    processed_rows = csv_update_df["row_id"].tolist()  # List of row_ids already processed
else:
    processed_rows = []  # No rows processed if the file doesn't exist yet

# Start processing from row 65 (or whichever row you want)
start_row = 0
qa_df = pd.read_csv("test_qa.csv")
with open("predictions_deep.txt", "a") as f:
    for idx, row in qa_df.iloc[start_row:].iterrows():  # Start from row 65 onward
        # Skip already processed rows
        if idx in processed_rows:
            continue

        question, dataset_name = row["question"], row["dataset"]
        output = process_question(question, dataset_name, idx)  # Pass idx as row_id

        if output:
            f.write(f"{output}\n")

        # Optionally, you can update your progress immediately in case you stop and resume
        update_csv(idx, output, "", question, dataset_name)