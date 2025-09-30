import pandas as pd
import numpy as np
import os
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Setup ---
# Define the input file created in Phase 2 and the output file for this phase
input_dir = "new_asthma_dataset"
input_filename = os.path.join(input_dir, "embedded_asthma_dataset.pkl")
output_filename = os.path.join(input_dir, "fuzzified_asthma_dataset.pkl")

# --- 2. Define the Fuzzy Universe and Membership Functions ---

# We will fuzzify the PEF reading as a percentage of the patient's personal best.
# This normalizes the data and aligns with clinical "zone" definitions.
# The universe of discourse will be from 0 to 110 (to allow for readings slightly above 100%).
pef_percentage = ctrl.Antecedent(np.arange(0, 111, 1), 'pef_percentage')

# Define the fuzzy membership functions based on the standard asthma action plan zones.
# We use triangular functions for a clear, intuitive model.
pef_percentage['red'] = fuzz.trimf(pef_percentage.universe, [0, 0, 50])
pef_percentage['yellow'] = fuzz.trimf(pef_percentage.universe, [45, 65, 85])
pef_percentage['green'] = fuzz.trimf(pef_percentage.universe, [80, 100, 110])

# --- 3. Visualize the Membership Functions (Optional but Recommended) ---
# This helps verify that your fuzzy sets correctly represent the clinical zones.
print("Visualizing the PEF Percentage fuzzy membership functions...")
pef_percentage.view()
plt.title("Fuzzy Sets for Asthma PEF Zones")
plt.show()

# --- 4. Define the Fuzzification Function ---
def fuzzify_pef_percentage(pef_percent):
    """
    Calculates the degree of membership for a single PEF percentage value
    across the 'red', 'yellow', and 'green' fuzzy sets.
    """
    memberships = {
        'red': fuzz.interp_membership(pef_percentage.universe, pef_percentage['red'].mf, pef_percent),
        'yellow': fuzz.interp_membership(pef_percentage.universe, pef_percentage['yellow'].mf, pef_percent),
        'green': fuzz.interp_membership(pef_percentage.universe, pef_percentage['green'].mf, pef_percent)
    }
    return memberships

# --- 5. Main Processing Logic ---
try:
    # Load the dataset with embeddings from Phase 2
    df = pd.read_pickle(input_filename)
    print(f"\nSuccessfully loaded '{input_filename}' with {len(df)} rows.")

    # First, calculate the PEF percentage for each row
    df['pef_percentage'] = (df['pef_reading'] / df['personal_best_pef']) * 100

    # Apply the fuzzification function to the new percentage column
    print("Applying fuzzification to PEF percentage values...")
    df['fuzzy_pef'] = df['pef_percentage'].apply(fuzzify_pef_percentage)

    # --- 6. Save the Fuzzified DataFrame ---
    df.to_pickle(output_filename)
    print(f"\nFuzzification complete. DataFrame saved to '{output_filename}'")

    # --- 7. Verification ---
    print("\n--- Verification of the output ---")
    # Display columns relevant to this phase to confirm correctness
    verification_columns = [
        'pef_reading',
        'personal_best_pef',
        'pef_percentage',
        'flare_state',
        'fuzzy_pef'
    ]
    print(df[verification_columns].head(10))
    
    # Example: Check a borderline case to see the fuzzy logic in action
    borderline_pef = 82.5
    print(f"\nExample of a borderline value ({borderline_pef}%):")
    print(fuzzify_pef_percentage(borderline_pef))

except FileNotFoundError:
    print(f"Error: The input file '{input_filename}' was not found.")
    print("Please make sure you have successfully run the Phase 2 script first.")
except Exception as e:
    print(f"An error occurred: {e}")
