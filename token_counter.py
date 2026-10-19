
import tiktoken

text = """
Large Language Models process text using tokens.
A context window limits how much tokenized input
and generated output can fit into a model invocation.
"""

# Example encoding commonly associated with GPT-4-class
# and GPT-3.5-class models. Not universal to every model.
encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode(text)

print("Original text:")
print(text)

print("\nToken IDs:")
print(tokens)

print("\nNumber of tokens:", len(tokens))

print("\nDecoded tokens:")
for token_id in tokens:
    print(repr(encoding.decode([token_id])))
