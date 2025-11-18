import spacy

print("Attempting to load 'en_core_web_sm'...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("Model 'en_core_web_sm' loaded successfully!")
    print(f"Pipeline components: {nlp.pipe_names}")
except Exception as e:
    print(f"Failed to load model. Error: {e}")
