import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
import gc

# --- 1. Setup ---
# Define the input file created in Phase 1 and the output directory
input_dir = "new_asthma_dataset"
input_filename = os.path.join(input_dir, "processed_asthma_dataset.csv")
output_filename = os.path.join(input_dir, "embedded_asthma_dataset.pkl")

# Use GPU if available for a significant speed-up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. Load Models ---
# Load spaCy model for efficient text cleaning (disable unused parts for speed)
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load HuggingFace sentence-transformer model and tokenizer
print("Loading HuggingFace model...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
model.eval()  # Set the model to evaluation mode (important for inference)

# --- 3. Define Processing Functions ---
def preprocess_text(text):
    """Converts text to lowercase, lemmatizes, and removes stopwords/punctuation."""
    # Ensure input is a string
    doc = nlp(str(text).lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def get_batch_embeddings(text_list):
    """Generates embeddings for a list of texts in a single batch for efficiency."""
    # Tokenize the sentences and send to the selected device
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Compute token embeddings without tracking gradients
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform mean pooling to get one vector per sentence
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Move embeddings back to CPU and convert to numpy
    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    return embeddings

# --- 4. Main Processing Logic ---
try:
    # Load the dataset created in Phase 1
    df = pd.read_csv(input_filename)
    print(f"\nSuccessfully loaded '{input_filename}' with {len(df)} rows.")

    # Apply text preprocessing to the 'report_text' column
    print("Preprocessing text reports...")
    df['processed_report'] = df['report_text'].apply(preprocess_text)

    # Generate embeddings for all reports in efficient batches
    print("Generating embeddings (this may take a moment)...")
    embeddings = get_batch_embeddings(df['processed_report'].tolist())
    df['embedding'] = list(embeddings) # Store embeddings as a list of arrays

    # --- 5. Save the Processed DataFrame ---
    # We use pickle (.pkl) to perfectly preserve the numpy array structure within the DataFrame
    df.to_pickle(output_filename)
    print(f"\nProcessing complete. DataFrame with embeddings saved to '{output_filename}'")

    # Clean up memory
    del df, embeddings
    gc.collect()

    # --- 6. Verification ---
    print("\n--- Verification of the output ---")
    # Load the saved file back to verify its contents
    verified_df = pd.read_pickle(output_filename)
    print("Sample of the final data:")
    print(verified_df[['report_text', 'processed_report', 'embedding']].head())
    print("\nShape of the first embedding vector:", verified_df['embedding'].iloc[0].shape)
    
except FileNotFoundError:
    print(f"Error: The input file '{input_filename}' was not found.")
    print("Please make sure you have successfully run the Phase 1 script first.")
