import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import spacy
from transformers import AutoTokenizer, AutoModel
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 1. Define the Neural Network Architecture ---
# This must be the EXACT same architecture as the one you trained and saved.
class FlareUpClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FlareUpClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# --- 2. The Master Predictor Class ---
class SymptomPredictor:
    def __init__(self, disease_type='diabetes'):
        """
        Initializes the predictor for a specific disease ('diabetes' or 'asthma').
        Loads all the necessary models and artifacts.
        """
        self.disease_type = disease_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing predictor for '{self.disease_type}'...")
        
        # --- Load NLP Models (shared by both) ---
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        self.embedding_model.eval()

        # --- Load Disease-Specific Artifacts ---
        artifacts_dir = f"artifacts/{self.disease_type}"
        if not os.path.exists(artifacts_dir):
            raise FileNotFoundError(f"Artifacts directory not found at '{artifacts_dir}'. Please ensure models are trained and saved.")
            
        print(f"Loading artifacts from '{artifacts_dir}'...")
        # Load scaler and label encoder
        self.scaler = joblib.load(os.path.join(artifacts_dir, f"{self.disease_type}_scaler.joblib"))
        self.encoder = joblib.load(os.path.join(artifacts_dir, f"{self.disease_type}_label_encoder.joblib"))
        
        # Load the trained PyTorch model
        input_size = self.scaler.n_features_in_
        num_classes = len(self.encoder.classes_)
        self.model = FlareUpClassifier(input_size, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(artifacts_dir, f"{self.disease_type}_classifier.pth")))
        self.model.eval()
        
        # --- Initialize the correct Fuzzy Inference System ---
        self._initialize_fis()
        print("Predictor initialized successfully.")

    def _initialize_fis(self):
        """Sets up the Fuzzy Inference System based on the disease type."""
        # Shared components for FIS
        symptom_severity = ctrl.Antecedent(np.arange(0, 11, 1), 'symptom_severity')
        symptom_severity['low'] = fuzz.trimf(symptom_severity.universe, [0, 0, 5])
        symptom_severity['high'] = fuzz.trimf(symptom_severity.universe, [5, 10, 10])
        
        risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')
        risk['low'] = fuzz.trimf(risk.universe, [0, 0, 5])
        risk['medium'] = fuzz.trimf(risk.universe, [2, 5, 8])
        risk['high'] = fuzz.trimf(risk.universe, [5, 10, 10])

        if self.disease_type == 'asthma':
            pef_percentage = ctrl.Antecedent(np.arange(0, 111, 1), 'pef_percentage')
            pef_percentage['red'] = fuzz.trimf(pef_percentage.universe, [0, 0, 50])
            pef_percentage['yellow'] = fuzz.trimf(pef_percentage.universe, [45, 65, 85])
            pef_percentage['green'] = fuzz.trimf(pef_percentage.universe, [80, 100, 110])
            
            rule1 = ctrl.Rule(pef_percentage['green'] & symptom_severity['low'], risk['low'])
            rule2 = ctrl.Rule(pef_percentage['yellow'], risk['medium'])
            rule3 = ctrl.Rule(pef_percentage['red'] | symptom_severity['high'], risk['high'])
            self.fis_control = ctrl.ControlSystem([rule1, rule2, rule3])

        # Add diabetes FIS logic here if you have it
        # For now, we'll focus on the completed asthma one.

    def _get_text_embedding(self, text):
        """Helper to preprocess and embed a single text string."""
        processed_text = " ".join([token.lemma_ for token in self.nlp(str(text).lower()) if not token.is_stop and not token.is_punct])
        encoded_input = self.tokenizer([processed_text], padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu().numpy()

    def analyze(self, symptom_text, clinical_value, personal_best_pef=None):
        """
        Analyzes a single patient report and returns a prediction and explanation.
        """
        # 1. Get Text Embedding
        embedding_vec = self._get_text_embedding(symptom_text)

        # 2. Get Fuzzy Vector
        if self.disease_type == 'asthma':
            if personal_best_pef is None:
                raise ValueError("`personal_best_pef` is required for asthma analysis.")
            pef_percent = (clinical_value / personal_best_pef) * 100
            fuzzy_vec = np.array([[
                fuzz.interp_membership(np.arange(0, 111, 1), fuzz.trimf(np.arange(0, 111, 1), [0, 0, 50]), pef_percent),
                fuzz.interp_membership(np.arange(0, 111, 1), fuzz.trimf(np.arange(0, 111, 1), [45, 65, 85]), pef_percent),
                fuzz.interp_membership(np.arange(0, 111, 1), fuzz.trimf(np.arange(0, 111, 1), [80, 100, 110]), pef_percent)
            ]])
        # Add diabetes fuzzy logic here
        
        # 3. Combine, Scale, and Predict
        combined_features = np.concatenate([embedding_vec, fuzzy_vec], axis=1)
        scaled_features = self.scaler.transform(combined_features)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            prediction = self.encoder.inverse_transform(predicted_idx.cpu().numpy())[0]

        # 4. Get FIS Explanation
        fis_sim = ctrl.ControlSystemSimulation(self.fis_control)
        if self.disease_type == 'asthma':
            # Simple severity score based on keywords
            severity_score = 9 if any(word in symptom_text.lower() for word in ['hard to breathe', 'struggling', 'can\'t speak']) else 3
            fis_sim.input['pef_percentage'] = pef_percent
            fis_sim.input['symptom_severity'] = severity_score
            fis_sim.compute()
            risk_score = fis_sim.output['risk']
            # **FIX APPLIED HERE**
            # Convert the consequents generator to a list before accessing it
            risk_terms = list(self.fis_control.consequents)[0].terms
            explanation_level = max(risk_terms, key=lambda term: fuzz.interp_membership(risk_terms[term].parent.universe, risk_terms[term].mf, risk_score))
            explanation = f"Risk is '{explanation_level.upper()}' due to PEF reading and symptom severity."
        
        return {"prediction": prediction, "explanation": explanation}

# --- 3. Example Usage ---
if __name__ == '__main__':
    try:
        # Initialize the predictor for Asthma
        asthma_predictor = SymptomPredictor(disease_type='asthma')
        
        print("\n--- Running Asthma Analysis ---")
        # Scenario 1: A high-risk asthma event
        symptom_report_1 = "It's very hard to breathe, even while I'm resting."
        pef_reading_1 = 210
        personal_best_1 = 500
        
        result_1 = asthma_predictor.analyze(symptom_report_1, pef_reading_1, personal_best_1)
        print(f"Report: '{symptom_report_1}' | PEF: {pef_reading_1}")
        print(f"--> Prediction: {result_1['prediction'].upper()}")
        print(f"--> Explanation: {result_1['explanation']}")
        
        print("\n" + "-"*20)
        
        # Scenario 2: A low-risk asthma event
        symptom_report_2 = "Feeling good, no issues at all today."
        pef_reading_2 = 480
        personal_best_2 = 500
        
        result_2 = asthma_predictor.analyze(symptom_report_2, pef_reading_2, personal_best_2)
        print(f"Report: '{symptom_report_2}' | PEF: {pef_reading_2}")
        print(f"--> Prediction: {result_2['prediction'].upper()}")
        print(f"--> Explanation: {result_2['explanation']}")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

