#!/usr/bin/env python3

import sys

current_key = None
current_tokens = []

for line in sys.stdin:
    line = line.strip()
    key, value = line.split('\t')
    value = eval(value)  # Convert the string representation of list back to a list

    if key != current_key:
        # Process the tokens for the previous key (abstract)
        if current_key is not None:
            cleaned_tokens = current_tokens
            # Join the cleaned tokens and enclose them in double quotes
            cleaned_tokens_str = '"' + ' '.join(cleaned_tokens) + '"'
            # Perform any additional processing or write to output here
            print(f"{current_key},{cleaned_tokens_str}")

        # Reset for the new key (abstract)
        current_key = key
        current_tokens = []

    # Append tokens to the current list
    current_tokens.extend(value)

# Process the last key (abstract)
if current_key is not None:
    cleaned_tokens = current_tokens
    # Join the cleaned tokens and enclose them in double quotes
    cleaned_tokens_str = '"' + ' '.join(cleaned_tokens) + '"'
    # Perform any additional processing or write to output here
    print(f"{current_key},{cleaned_tokens_str}")
