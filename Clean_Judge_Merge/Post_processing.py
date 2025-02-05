import re

# Read the content of the file
with open("predictions_lite.txt", "r") as file:
    content = file.read()

# Regular expression pattern to match np.uint8(*), np.int64(*), etc.
pattern = r"np\.(?:uint8|int64|float64)\((.*?)\)"

# Replace matches with just the inner value
modified_content = re.sub(pattern, r"\1", content)

# Write the modified content back to the file
with open("predictions_lite.txt", "w") as file:
    file.write(modified_content)

print("Replacements done!")
