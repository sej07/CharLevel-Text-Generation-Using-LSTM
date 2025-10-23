import urllib.request
import os

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = "data/input.txt"
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return 
    print("Downloading dataset")
    urllib.request.urlretrieve(url, output_path)
    print(f"Successfully downloaded to {output_path}")
    with open(output_path, 'r', encoding='utf-8') as f:
        text = f.read()
        print(f"\n Dataset Statistics:")
        print(f"  - Total characters: {len(text):,}")
        print(f"  - Unique characters: {len(set(text)):,}")
        print(f"\nFirst 300 characters:")
        print("\n")
        print(text[:300])

if __name__== "__main__":
    download_shakespeare()
