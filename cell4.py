import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import gc

# --- 1. Setup ---
# Define the input file from Phase 3 and the output directory for our new artifacts
input_dir = "new_asthma_dataset"
input_filename = os.path.join(input_dir, "fuzzified_asthma_dataset.pkl")
output_dir_artifacts = "artifacts/asthma"
os.makedirs(output_dir_artifacts, exist_ok=True)

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. Load and Prepare Data ---
try:
    df = pd.read_pickle(input_filename)
    print(f"Successfully loaded '{input_filename}' with {len(df)} rows.")

    # a) Process Fuzzy Features from the 'fuzzy_pef' column
    fuzzy_features = pd.DataFrame(df['fuzzy_pef'].tolist()).to_numpy()

    # b) Process Embedding Features
    embedding_features = np.vstack(df['embedding'].values)
    
    # c) Combine into the final feature matrix X
    X = np.concatenate([embedding_features, fuzzy_features], axis=1)

    # d) Prepare and encode the 'flare_state' labels
    y_labels = df['flare_state'].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    
    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Label vector y shape: {y.shape}")
    print(f"Classes found and encoded: {encoder.classes_}") # E.g., ['green' 'red' 'yellow']

    # Clean up memory
    del df
    gc.collect()

    # --- 3. Split and Scale the Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features for optimal neural network performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Create PyTorch DataLoaders ---
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # --- 5. Define the Neural Network Model ---
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
            x = self.layer3(x) # No softmax here, as CrossEntropyLoss includes it
            return x

    # --- 6. Training and Evaluation ---
    input_size = X.shape[1]
    num_classes = len(encoder.classes_)
    
    model = FlareUpClassifier(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 25 # A few more epochs can be beneficial
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"\nTraining finished.")
    print(f"Accuracy on the test set: {100 * correct / total:.2f} %")

    # --- 7. Save the Asthma-Specific Artifacts ---
    print("\nSaving model and data processors...")
    # Save the trained model's state
    torch.save(model.state_dict(), os.path.join(output_dir_artifacts, "asthma_classifier.pth"))
    # Save the fitted scaler
    joblib.dump(scaler, os.path.join(output_dir_artifacts, "asthma_scaler.joblib"))
    # Save the fitted label encoder
    joblib.dump(encoder, os.path.join(output_dir_artifacts, "asthma_label_encoder.joblib"))
    print(f"All artifacts saved successfully to the '{output_dir_artifacts}' directory.")

except FileNotFoundError:
    print(f"Error: The input file '{input_filename}' was not found.")
    print("Please make sure you have successfully run the Phase 3 script first.")
except Exception as e:
    print(f"An error occurred: {e}")
