import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Constants
RANDOM_STATE = 42
K_FOLDS = 5
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./bert_results"
METRIC_FILE = "bert_metrics.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path='data_filtered.csv'):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    return df

def tokenize_function(example, tokenizer):
    """Tokenize text data"""
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_and_evaluate_fold(train_df, test_df, learning_rate, num_epochs, tokenizer):
    """Train and evaluate model on a single fold"""
    # Convert pandas DataFrames into Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_tokenized = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    
    # Set the format for PyTorch
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Set up training arguments
    fold_output_dir = os.path.join(OUTPUT_DIR, f"lr_{learning_rate}_epochs_{num_epochs}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(fold_output_dir, 'logs'),
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    
    return eval_results

def k_fold_cross_validation(df, k=K_FOLDS):
    """Perform k-fold cross-validation"""
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    
    # List of learning rates and epochs to try
    learning_rates = [2e-5, 5e-5, 1e-4]
    num_epochs_list = [2, 3, 4]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Store results for hyperparameter tuning
    hyperparam_results = []
    
    # Store metrics across folds for the best hyperparameter combination
    best_lr = None
    best_epochs = None
    best_avg_acc = 0
    
    # Hyperparameter tuning on first fold only to save compute time
    print("Performing hyperparameter tuning on first fold...")
    
    # Get the first fold
    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        
        for lr in learning_rates:
            for epochs in num_epochs_list:
                print(f"\nTraining with lr={lr}, epochs={epochs}")
                eval_results = train_and_evaluate_fold(train_df, test_df, lr, epochs, tokenizer)
                
                # Fix: Extract metrics with correct keys
                # The eval_results keys have 'eval_' prefix, so we need to handle that
                accuracy = eval_results.get('eval_accuracy', 0)
                f1 = eval_results.get('eval_f1', 0)
                precision = eval_results.get('eval_precision', 0)
                recall = eval_results.get('eval_recall', 0)
                
                # Store results
                hyperparam_results.append({
                    'learning_rate': lr,
                    'epochs': epochs,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                })
                
                # Update best hyperparameters if needed
                if accuracy > best_avg_acc:
                    best_avg_acc = accuracy
                    best_lr = lr
                    best_epochs = epochs
        
        # Only use the first fold for hyperparameter tuning
        break
    
    # Plot hyperparameter tuning results
    plot_hyperparameter_tuning(hyperparam_results)
    
    # Run k-fold CV with the best hyperparameters
    print(f"\nRunning {k}-fold cross-validation with best hyperparameters: lr={best_lr}, epochs={best_epochs}")
    
    # Initialize arrays to store metrics across folds
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    fold_results = []
    
    # Perform k-fold cross-validation with best hyperparameters
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\nTraining fold {fold + 1}/{k}")
        
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        
        # Train and evaluate on this fold
        eval_results = train_and_evaluate_fold(train_df, test_df, best_lr, best_epochs, tokenizer)
        
        # Fix: Extract metrics with correct keys
        accuracy = eval_results.get('eval_accuracy', 0)
        f1 = eval_results.get('eval_f1', 0)
        precision = eval_results.get('eval_precision', 0)
        recall = eval_results.get('eval_recall', 0)
        
        # Store metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"Fold {fold + 1} results: accuracy={accuracy:.3f}, f1={f1:.3f}, precision={precision:.3f}, recall={recall:.3f}")
    
    # Calculate means and standard deviations
    metrics = {
        'Accuracy': (np.mean(accuracies), np.std(accuracies)),
        'Precision': (np.mean(precisions), np.std(precisions)),
        'Recall': (np.mean(recalls), np.std(recalls)),
        'F1 Score': (np.mean(f1_scores), np.std(f1_scores))
    }
    
    print("\nFinal BERT metrics:")
    for metric, (mean, std) in metrics.items():
        print(f"{metric}: {mean:.3f} ± {std:.3f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(fold_results)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, METRIC_FILE), index=False)
    
    # Plot fold results
    plot_fold_results(fold_results)
    
    return metrics, hyperparam_results

def plot_hyperparameter_tuning(results):
    """Plot hyperparameter tuning results"""
    if not results:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Plot accuracy vs learning rate for different epochs
    for i, epochs in enumerate(sorted(df['epochs'].unique())):
        subset = df[df['epochs'] == epochs]
        axs[0].plot(subset['learning_rate'], subset['accuracy'], 'o-', 
                   label=f'{epochs} epochs', markersize=8)
    
    axs[0].set_title('Accuracy vs Learning Rate')
    axs[0].set_xlabel('Learning Rate')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xscale('log')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot F1 vs learning rate for different epochs
    for i, epochs in enumerate(sorted(df['epochs'].unique())):
        subset = df[df['epochs'] == epochs]
        axs[1].plot(subset['learning_rate'], subset['f1'], 'o-', 
                   label=f'{epochs} epochs', markersize=8)
    
    axs[1].set_title('F1 Score vs Learning Rate')
    axs[1].set_xlabel('Learning Rate')
    axs[1].set_ylabel('F1 Score')
    axs[1].set_xscale('log')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot accuracy vs epochs for different learning rates
    for i, lr in enumerate(sorted(df['learning_rate'].unique())):
        subset = df[df['learning_rate'] == lr]
        axs[2].plot(subset['epochs'], subset['accuracy'], 'o-', 
                   label=f'LR: {lr}', markersize=8)
    
    axs[2].set_title('Accuracy vs Number of Epochs')
    axs[2].set_xlabel('Number of Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Plot F1 vs epochs for different learning rates
    for i, lr in enumerate(sorted(df['learning_rate'].unique())):
        subset = df[df['learning_rate'] == lr]
        axs[3].plot(subset['epochs'], subset['f1'], 'o-', 
                   label=f'LR: {lr}', markersize=8)
    
    axs[3].set_title('F1 Score vs Number of Epochs')
    axs[3].set_xlabel('Number of Epochs')
    axs[3].set_ylabel('F1 Score')
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperparameter_tuning.png'))
    plt.close()

def plot_fold_results(fold_results):
    """Plot metrics across folds"""
    df = pd.DataFrame(fold_results)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.plot(df['fold'], df[metric], 'o-', label=metric.capitalize(), markersize=8)
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Performance Metrics Across Folds')
    plt.xticks(df['fold'])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fold_results.png'))
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Perform k-fold cross-validation
    print(f"Performing {K_FOLDS}-fold cross-validation...")
    metrics, hyperparam_results = k_fold_cross_validation(df, k=K_FOLDS)
    
    # Print final metrics
    print("\nFinal BERT metrics:")
    for metric, (mean, std) in metrics.items():
        print(f"{metric}: {mean:.3f} ± {std:.3f}")
    
    # Find best hyperparameters
    best_result = max(hyperparam_results, key=lambda x: x['accuracy'])
    print(f"\nBest hyperparameters: lr={best_result['learning_rate']}, epochs={best_result['epochs']}")
    print(f"Best validation accuracy: {best_result['accuracy']:.3f}")
    
    print("\nDone! Results saved to CSV and PNG files.")

if __name__ == "__main__":
    main()