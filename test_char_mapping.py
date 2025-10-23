with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Total characters: {len(text)}")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique Characters: {vocab_size}")
print(f"Characters: {chars}")
char_to_idx = {ch: i for i , ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
print("\n--- Testing Mapping ---")
test_word = "Hello"
print(f"Word: '{test_word}'")

indices = [char_to_idx[ch] for ch in test_word]
print(f"Indices: {indices}")

# Convert back to characters
decoded = ''.join([idx_to_char[i] for i in indices])
print(f"Decoded: '{decoded}'")

print("\nMapping works!" if decoded == test_word else "ERROR!")