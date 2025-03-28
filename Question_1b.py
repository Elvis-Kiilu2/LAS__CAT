# Simple English-Swahili dictionary
eng_to_swa = {
    "hello": "habari",
    "goodbye": "kwa heri",
    "thank you": "asante",
    "how are you": "habari yako",
    "i love you": "nakupenda",
    "good morning": "habari ya asubuhi"
}

swa_to_eng = {v: k for k, v in eng_to_swa.items()}

def translate_english_to_swahili(text):
    text = text.lower()
    return eng_to_swa.get(text, "Translation not found")

def translate_swahili_to_english(text):
    text = text.lower()
    return swa_to_eng.get(text, "Translation not found")

# Test the translation with examples
def test_translation():
    # English to Swahili
    print("English to Swahili:")
    examples_eng = ["hello", "thank you", "i love you", "good morning"]
    for phrase in examples_eng:
        translation = translate_english_to_swahili(phrase)
        print(f"{phrase} -> {translation}")

    # Swahili to English
    print("\nSwahili to English:")
    examples_swa = ["habari", "asante", "nakupenda", "kwa heri"]
    for phrase in examples_swa:
        translation = translate_swahili_to_english(phrase)
        print(f"{phrase} -> {translation}")

# Run the test
test_translation()