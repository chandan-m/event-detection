import re

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = list(text.split())
    new_tokens = list()
    for token in tokens:
        if len(token) <= 1: continue
        new_tokens.append(token)
    return new_tokens
