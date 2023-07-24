import re

def check_keys_in_text(keys, text):
    # Preprocess keys to generate regular expression patterns
    regex_patterns = ['.*'.join(key) for key in keys]

    for pattern in regex_patterns:
        # Construct regex that matches zero or more spaces or hyphens between characters
        regex = re.compile("[-\s]*".join(pattern))
        if regex.search(text):
            return True

    return False

keys = ['hello', 'world']
text = 'h e-l-l- o w o-rld'
print(check_keys_in_text(keys, text))  # It should print True

