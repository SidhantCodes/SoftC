import pandas as pd
import numpy as np
import os
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Setup and Load Sample Data ---
# We only need a sample of the data to get the context for our rules.
input_dir = "new_asthma_dataset"
input_filename = os.path.join(input_dir, "fuzzified_asthma_dataset.pkl")

try:
    df = pd.read_pickle(input_filename)
    print(f"Successfully loaded '{input_filename}' to build the FIS.")
except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    exit()

# --- 2. Define Fuzzy Inputs and Outputs ---

# Input 1: PEF Percentage (reusing the logic from Phase 3)
pef_percentage = ctrl.Antecedent(np.arange(0, 111, 1), 'pef_percentage')
pef_percentage['red'] = fuzz.trimf(pef_percentage.universe, [0, 0, 50])
pef_percentage['yellow'] = fuzz.trimf(pef_percentage.universe, [45, 65, 85])
pef_percentage['green'] = fuzz.trimf(pef_percentage.universe, [80, 100, 110])

# Input 2: Symptom Severity
# We'll create a simple numerical score (0-10) to represent the severity
# inferred from the text reports.
symptom_severity = ctrl.Antecedent(np.arange(0, 11, 1), 'symptom_severity')
symptom_severity['low'] = fuzz.trimf(symptom_severity.universe, [0, 0, 5])
symptom_severity['high'] = fuzz.trimf(symptom_severity.universe, [5, 10, 10])

# Output: Risk Level
risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 5])
risk['medium'] = fuzz.trimf(risk.universe, [2, 5, 8])
risk['high'] = fuzz.trimf(risk.universe, [5, 10, 10])

# Visualize the new variables
symptom_severity.view()
plt.title("Fuzzy Sets for Symptom Severity")
risk.view()
plt.title("Fuzzy Sets for Risk Level")
plt.show()

# --- 3. Define the Fuzzy Rules ---
# These are the "common sense" rules that mimic a clinician's logic.
rule1 = ctrl.Rule(pef_percentage['green'] & symptom_severity['low'], risk['low'])
rule2 = ctrl.Rule(pef_percentage['yellow'] & symptom_severity['low'], risk['medium'])
rule3 = ctrl.Rule(pef_percentage['yellow'] & symptom_severity['high'], risk['medium'])
rule4 = ctrl.Rule(pef_percentage['red'], risk['high']) # A red PEF reading is high risk regardless of symptoms
rule5 = ctrl.Rule(symptom_severity['high'], risk['high']) # Severe symptoms are high risk regardless of PEF

# --- 4. Create the Control System ---
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

# --- 5. Test the FIS with an Example Scenario ---
# Let's simulate a patient with a PEF reading at 60% of their best (Yellow Zone)
# and a high symptom severity score.
test_pef_percent = 60

# We need a way to get a severity score from text. For now, we'll simulate it.
# A real application would have a simple model or keyword search for this.
test_symptom_text = "My rescue inhaler isn't helping much." 
# Let's assign a high severity score for this text.
test_severity_score = 9

# Pass the inputs to the simulation
risk_simulation.input['pef_percentage'] = test_pef_percent
risk_simulation.input['symptom_severity'] = test_severity_score

# Compute the result
risk_simulation.compute()

# --- 6. Print the Explanation ---
print("\n--- Example FIS Explanation ---")
print(f"Input PEF Percentage: {test_pef_percent}%")
print(f"Input Symptom Severity Score: {test_severity_score}")
print(f"Predicted Risk Score: {risk_simulation.output['risk']:.2f}")

# Find the dominant linguistic term for the output
risk_level = max(risk.terms, key=lambda term: fuzz.interp_membership(risk.universe, risk[term].mf, risk_simulation.output['risk']))
print(f"Conclusion: The risk is '{risk_level.upper()}' because the PEF is in the 'yellow' zone and symptom reports are severe.")

# You can also visualize the reasoning
risk.view(sim=risk_simulation)
plt.show()

