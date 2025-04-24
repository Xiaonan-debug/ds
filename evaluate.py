import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Number of folds for cross-validation
K_FOLDS = 5
# Random state for reproducibility
RANDOM_STATE = 42

def load_data(file_path='data_filtered.csv'):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    # Separate features and labels
    X = df['text']
    y = df['label']
    return X, y

def create_classifiers():
    """Create a dictionary of classifier pipelines to evaluate"""
    classifiers = {
        "Random Uniform": Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', DummyClassifier(strategy='uniform', random_state=RANDOM_STATE))
        ]),
        "Majority Class": Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', DummyClassifier(strategy='most_frequent'))
        ]),
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=RANDOM_STATE))
        ]),
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', MultinomialNB(alpha=0.001))
        ]),
        "Random Forest": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
            ('clf', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE))
        ]),
        "KNN": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
            ('clf', KNeighborsClassifier(n_neighbors=9, weights='uniform'))
        ])
    }
    return classifiers

def evaluate_auc_with_cv(X, y):
    """Train models using k-fold cross-validation and evaluate AUC scores"""
    # Create classifiers
    classifiers = create_classifiers()
    
    # Create a KFold object
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Dictionary to store results
    results = {}
    
    # For each classifier
    for name, model in classifiers.items():
        print(f"\nEvaluating {name}...")
        
        # Lists to store fold results
        fold_auc_scores = []
        all_fpr = []
        all_tpr = []
        all_y_true = []
        all_y_prob = []
        
        # Perform k-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train the model on this fold
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba
                y_prob = model.decision_function(X_test)
            
            # Calculate AUC for this fold
            fold_auc = roc_auc_score(y_test, y_prob)
            fold_auc_scores.append(fold_auc)
            
            # Calculate ROC curve for this fold
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            
            # Store true labels and predicted probabilities
            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
            
            print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")
        
        # Calculate mean AUC across folds
        mean_auc = np.mean(fold_auc_scores)
        std_auc = np.std(fold_auc_scores)
        
        # Calculate overall AUC on combined predictions
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
        
        print(f"  Mean AUC: {mean_auc:.4f} (±{std_auc:.4f})")
        print(f"  Overall AUC: {overall_auc:.4f}")
        
        # Calculate overall ROC curve
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        
        # Store results
        results[name] = {
            'auc': overall_auc,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'fpr': fpr,
            'tpr': tpr
        }
    
    return results

def plot_roc_curves(results):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    # Line colors for different models
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    
    # Sort models by AUC score (descending)
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['auc'], reverse=True)
    
    # Plot ROC curve for each model
    for i, name in enumerate(sorted_models):
        result = results[name]
        plt.plot(result['fpr'], result['tpr'], color=colors[i % len(colors)], lw=2,
                 label=f'{name} (AUC = {result["auc"]:.4f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()
    
    # Create a bar chart for AUC scores
    plt.figure(figsize=(12, 6))
    
    # Extract AUC scores and sort
    models = [name for name in sorted_models]
    auc_scores = [results[name]['auc'] for name in sorted_models]
    
    # Plot bars
    bars = plt.bar(models, auc_scores, color='skyblue')
    
    # Add value labels and error bars
    for i, (bar, model) in enumerate(zip(bars, models)):
        height = bar.get_height()
        std_dev = results[model]['std_auc']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add error bar
        plt.errorbar(i, height, yerr=std_dev, color='black', capsize=5, fmt='none')
    
    # Set plot details
    plt.ylim([0.5, 1.0])  # Start from 0.5 since that's random chance
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores Comparison (with Standard Deviation)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('auc_scores.png')
    plt.close()
    
    # Return AUC scores for the table
    return {name: (results[name]['auc'], results[name]['mean_auc'], results[name]['std_auc']) 
            for name in sorted_models}

def create_auc_table(auc_scores):
    """Create a table of AUC scores"""
    # Create DataFrame
    data = [(name, scores[0], scores[1], scores[2]) 
            for name, scores in auc_scores.items()]
    df = pd.DataFrame(data, columns=['Model', 'Overall AUC', 'Mean AUC', 'Std Dev'])
    
    # Sort by Overall AUC (descending)
    df = df.sort_values('Overall AUC', ascending=False)
    
    # Save to CSV
    df.to_csv('auc_scores.csv', index=False)
    
    return df

def main():
    # Load data
    print("Loading data...")
    X, y = load_data()
    
    # Train models and evaluate AUC
    print("Training and evaluating models...")
    results = evaluate_auc_with_cv(X, y)
    
    # Plot ROC curves
    print("Plotting ROC curves...")
    auc_scores = plot_roc_curves(results)
    
    # Create AUC score table
    print("Creating AUC score table...")
    auc_table = create_auc_table(auc_scores)
    
    # Print results
    print("\nAUC Scores:")
    print(auc_table)
    
    print("\nDone! Results saved to CSV and PNG files.")

if __name__ == "__main__":
    main()