import ast
import numpy as np
import re

# Function to identify answer type and format it
def format_answer(answer):
    answer = answer.strip()

    # Handle empty lists explicitly
    if answer == "[]":
        return []

    # Boolean Type
    if answer.lower() in {"true", "false", "yes", "no", "y", "n"}:
        return answer.capitalize() if answer.lower() in {"true", "false"} else ("Yes" if answer.lower() in {"yes", "y"} else "No")

    # Number Type (integer or float)
    if re.match(r"^-?\d+(\.\d+)?$", answer):
        return float(answer) if "." in answer else int(answer)

    # Handle lists safely using ast.literal_eval
    if answer.startswith("[") and answer.endswith("]"):
        try:
            parsed_list = ast.literal_eval(answer)

            # Convert numpy types to native Python types
            def convert_np(val):
                if isinstance(val, (np.integer, np.floating)):
                    return val.item()
                elif isinstance(val, str) and "np." in val:
                    return eval(val).item()  # Evaluate numpy type expressions
                return val

            parsed_list = [convert_np(val) for val in parsed_list]

            # Ensure all elements are numbers and return the list
            return [x for x in parsed_list if isinstance(x, (int, float))]
        except (SyntaxError, ValueError, NameError):
            pass  # Fall through to category handling

    # Dictionary Type (convert to string for now)
    if answer.startswith("{") and answer.endswith("}"):
        return answer  # Keep dictionary as a string (adjust based on requirements)

    # Category Type (single string)
    return answer

# Read the input file and process each line
input_file = "predictions_lite.txt"
output_file = "formatted_predictions.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        formatted_answer = format_answer(line.strip())
        outfile.write(str(formatted_answer) + "\n")

print(f"Formatted predictions saved to '{output_file}'.")
