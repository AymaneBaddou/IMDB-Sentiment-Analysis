import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION & SETUP ---
DATA_PATH = 'IMDB Dataset.csv'
MODEL_DIR = 'model'
RESULTS_DIR = 'results'

# Create the folder structure
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# --- STEP 1: PREPROCESSING  ---
def clean_text(text):
    """
    Performs basic cleaning: lowercasing
    """
    text = str(text).lower()  # Lowercase
    return text

# --- STEP 2: FEATURE ENGINEERING  ---
def get_features(text):
    """
    Extracts sentence-level polarity features.
    Forbidden: Word-level scores or padded vectors.
    """
    # 1. VADER Polarity
    vader = analyzer.polarity_scores(text)
    
    # 2. TextBlob Polarity
    blob = TextBlob(text)
    
    return [
        vader['neg'],
        vader['neu'],
        vader['pos'],
        vader['compound'],
        blob.sentiment.polarity,
        blob.sentiment.subjectivity
    ]

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    # EDA Snippet 
    print(f"Data Loaded: {df.shape}")
    print("Class Distribution:\n", df['sentiment'].value_counts())

    # 2. Preprocessing & Feature Engineering
    print("Preprocessing and extracting features (this takes a moment)...")
    
    # Clean text first
    df['clean_review'] = df['review'].apply(clean_text)
    
    # Extract features from cleaned text
    X = np.array([get_features(text) for text in df['clean_review']])
    
    # Label Encoding (positive=1, negative=0) 
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

    # 3. Train / Validation / Test Split 
    # First split: 80% Train+Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Second split: Split the 80% into Train (80% of total) and Val (10% of total) roughly
    # splitting X_temp (which is 80%) into 0.875/0.125 gives approx 70/10/20 overall, 
    # but let's do a simple 0.25 split of X_temp to get Validation.
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Feature Scaling 
    # Although VADER/TextBlob are -1 to 1, scaling helps MLP convergence.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #5. Model Training
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        activation='relu',           
        solver='adam',
        max_iter=1000,                
        random_state=42,
        early_stopping=True,         
        validation_fraction=0.1,
        verbose=True
    )

    print("Training MLP Model...")
    clf.fit(X_train_scaled, y_train)

    # 6. Evaluation 
    print("Evaluating on Test Set...")
    # Predict using threshold 0.5 (default in sklearn predict) 
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")

    
    # Save Model
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save Metrics
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test Accuracy: {acc}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"✅ Metrics saved to {metrics_path}")

    # Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()
    print(f"✅ Confusion Matrix saved to {RESULTS_DIR}")

    # Save Loss Curve 
    plt.figure(figsize=(6, 5))
    plt.plot(clf.loss_curve_)
    plt.title('Binary Cross-Entropy Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curves.png'))
    plt.close()
    print(f"✅ Loss Curve saved to {RESULTS_DIR}")

    # Save Submission CSV
    submission_df = pd.DataFrame({
        'id': range(len(y_pred)),
        'sentiment': y_pred
    })
    submission_df.to_csv('submission.csv', index=False)
    print("✅ submission.csv created.")

if __name__ == "__main__":
    main()