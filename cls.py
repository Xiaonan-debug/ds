import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
# Additional classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

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

def train_evaluate_model(model, X, y, k_folds=K_FOLDS):
    """Train and evaluate model using k-fold cross-validation"""
    # Initialize arrays to store metrics for each fold
    accuracies = np.zeros(k_folds)
    precisions = np.zeros(k_folds)
    recalls = np.zeros(k_folds)
    f1_scores = np.zeros(k_folds)
    
    # Initialize KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracies[i] = accuracy_score(y_test, y_pred)
        precisions[i] = precision_score(y_test, y_pred, average='binary')
        recalls[i] = recall_score(y_test, y_pred, average='binary')
        f1_scores[i] = f1_score(y_test, y_pred, average='binary')
    
    # Calculate mean and standard deviation for each metric
    metrics = {
        'Accuracy': (np.mean(accuracies), np.std(accuracies)),
        'Precision': (np.mean(precisions), np.std(precisions)),
        'Recall': (np.mean(recalls), np.std(recalls)),
        'F1 Score': (np.mean(f1_scores), np.std(f1_scores))
    }
    
    return metrics

def hyperparameter_tuning(X, y, model_type="logreg"):
    """Perform hyperparameter tuning for different classifiers"""
    # Define the pipeline with TF-IDF vectorizer
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', None)  # Placeholder, will be set based on model_type
    ])
    
    # Define parameter grid based on model type
    if model_type == "logreg":
        pipeline.set_params(clf=LogisticRegression(random_state=RANDOM_STATE))
        param_grid = {
            'tfidf__max_features': [None, 5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear']
        }
        title = "Logistic Regression"
    
    elif model_type == "svm":
        pipeline.set_params(clf=SVC(random_state=RANDOM_STATE))
        param_grid = {
            'tfidf__max_features': [None],
            'tfidf__ngram_range': [(1, 1)],
            'clf__C': [0.1, 1.0, 10.0, 100],
            'clf__kernel': ['linear']
        }
        title = "Support Vector Machine"
    
    elif model_type == "rf":
        pipeline.set_params(clf=RandomForestClassifier(random_state=RANDOM_STATE))
        param_grid = {
            'tfidf__max_features': [5000],
            'tfidf__ngram_range': [(1, 1)],
            'clf__n_estimators': [100, 200, 300, 400],
            'clf__max_depth': [None, 10, 20]
        }
        title = "Random Forest"
    
    elif model_type == "gb":
        pipeline.set_params(clf=GradientBoostingClassifier(random_state=RANDOM_STATE))
        param_grid = {
            'tfidf__max_features': [5000],
            'tfidf__ngram_range': [(1, 1)],
            'clf__n_estimators': [100, 200, 300, 400],
            'clf__learning_rate': [0.1],
            'clf__max_depth': [5]
        }
        title = "Gradient Boosting"
        
    elif model_type == "nb":
        pipeline.set_params(clf=MultinomialNB())
        param_grid = {
            'tfidf__max_features': [None, 5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        }
        title = "Naive Bayes"
        
    elif model_type == "knn":
        pipeline.set_params(clf=KNeighborsClassifier())
        param_grid = {
            'tfidf__max_features': [5000],
            'tfidf__ngram_range': [(1, 1)],
            'clf__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'clf__weights': ['uniform']
        }
        title = "K-Nearest Neighbors"
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # Use 3-fold CV for faster hyperparameter tuning
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Split data for hyperparameter optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Fit the grid search to the data
    print(f"\nPerforming grid search for {title}...")
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters for {title}:", grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Evaluate the best model on the validation set
    best_model = grid_search.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.3f}")
    
    # Plot hyperparameter tuning results
    plot_hyperparameter_tuning(grid_search, model_type)
    
    return best_model

def plot_hyperparameter_tuning(grid_search, model_type):
    """Plot hyperparameter tuning results"""
    plt.figure(figsize=(10, 6))
    
    # Extract parameter values and scores
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Different plots depending on model type
    if model_type == "logreg":
        # Extract results for l2 penalty only to simplify the plot
        l2_results = results[results['param_clf__penalty'] == 'l2']
        
        # Group by C values and calculate mean scores
        c_values = sorted(l2_results['param_clf__C'].unique())
        mean_scores = [l2_results[l2_results['param_clf__C'] == c]['mean_test_score'].mean() for c in c_values]
        
        plt.plot(c_values, mean_scores, 'o-', markersize=8)
        plt.xscale('log')
        plt.xlabel('Regularization Parameter (C)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Logistic Regression Hyperparameter Tuning (L2 Penalty)')
    
    elif model_type == "svm":
        # Group by kernel and C value
        for kernel in results['param_clf__kernel'].unique():
            kernel_results = results[results['param_clf__kernel'] == kernel]
            c_values = sorted(kernel_results['param_clf__C'].unique())
            mean_scores = [kernel_results[kernel_results['param_clf__C'] == c]['mean_test_score'].mean() for c in c_values]
            
            plt.plot(c_values, mean_scores, 'o-', markersize=8, label=f'Kernel: {kernel}')
        
        plt.xscale('log')
        plt.xlabel('Regularization Parameter (C)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('SVM Hyperparameter Tuning')
        plt.legend()
    
    elif model_type in ["rf", "gb"]:
        # Group by n_estimators and max_depth
        parameter_name = 'n_estimators'
        for depth in results['param_clf__max_depth'].unique():
            depth_results = results[results['param_clf__max_depth'] == depth]
            estimator_values = sorted(depth_results['param_clf__n_estimators'].unique())
            mean_scores = [depth_results[depth_results['param_clf__n_estimators'] == n]['mean_test_score'].mean() 
                         for n in estimator_values]
            
            plt.plot(estimator_values, mean_scores, 'o-', markersize=8, 
                   label=f'Max Depth: {depth if depth is not None else "None"}')
        
        plt.xlabel('Number of Estimators')
        plt.ylabel('Cross-Validation Accuracy')
        model_name = "Random Forest" if model_type == "rf" else "Gradient Boosting"
        plt.title(f'{model_name} Hyperparameter Tuning')
        plt.legend()
    
    elif model_type == "nb":
        # Group by alpha
        alpha_values = sorted(results['param_clf__alpha'].unique())
        mean_scores = [results[results['param_clf__alpha'] == a]['mean_test_score'].mean() for a in alpha_values]
        
        plt.plot(alpha_values, mean_scores, 'o-', markersize=8)
        plt.xscale('log')
        plt.xlabel('Alpha (Smoothing Parameter)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Naive Bayes Hyperparameter Tuning')
    
    elif model_type == "knn":
        # Group by n_neighbors
        k_values = sorted(results['param_clf__n_neighbors'].unique())
        
        for weights in results['param_clf__weights'].unique():
            weights_results = results[results['param_clf__weights'] == weights]
            mean_scores = [weights_results[weights_results['param_clf__n_neighbors'] == k]['mean_test_score'].mean() 
                         for k in k_values]
            
            plt.plot(k_values, mean_scores, 'o-', markersize=8, label=f'Weights: {weights}')
        
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN Hyperparameter Tuning')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_type}_hyperparameter_tuning.png')
    plt.close()

def create_results_table(model_metrics):
    """Create a table with model metrics (mean ± std)"""
    # Format each cell as "mean ± std"
    formatted_metrics = {}
    for model_name, metrics in model_metrics.items():
        formatted_metrics[model_name] = {}
        for metric_name, (mean, std) in metrics.items():
            formatted_metrics[model_name][metric_name] = f"{mean:.3f} ± {std:.3f}"
    
    # Create DataFrame for the table
    df = pd.DataFrame(formatted_metrics).T
    
    # Save to CSV
    df.to_csv('model_metrics.csv')
    
    # Plot table as a figure
    fig, ax = plt.figure(figsize=(12, len(model_metrics) + 1)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Model Performance Metrics (mean ± std)')
    plt.tight_layout()
    plt.savefig('model_metrics_table.png')
    
    return df

def plot_metrics_comparison(model_metrics):
    """Plot comparison of model metrics with error bars"""
    models = list(model_metrics.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Create figure with 4 subplots (one for each metric)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        means = [model_metrics[model][metric][0] for model in models]
        stds = [model_metrics[model][metric][1] for model in models]
        
        axs[i].bar(np.arange(len(models)), means, yerr=stds, capsize=10)
        axs[i].set_title(f'{metric} Comparison')
        axs[i].set_xticks(np.arange(len(models)))
        axs[i].set_xticklabels(models, rotation=45, ha='right')
        axs[i].set_ylim(0, 1)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate bars with values
        for j, (mean, std) in enumerate(zip(means, stds)):
            axs[i].text(j, mean + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    
    # Create a heatmap of model performance
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(metrics), len(models)))
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            heatmap_data[i, j] = model_metrics[model][metric][0]  # Only mean values
    
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
               xticklabels=models, yticklabels=metrics, vmin=0, vmax=1)
    plt.title('Model Performance Heatmap')
    plt.tight_layout()
    plt.savefig('model_performance_heatmap.png')

def main():
    # Load data
    X, y = load_data()
    
    # Create and evaluate models with k-fold cross-validation
    model_metrics = {}
    
    # 1. Dummy classifier (random uniform)
    # print("\nEvaluating random uniform baseline...")
    # dummy_uniform = Pipeline([
    #     ('tfidf', TfidfVectorizer()),  # Dummy transform that will be ignored
    #     ('clf', DummyClassifier(strategy='uniform', random_state=RANDOM_STATE))
    # ])
    # model_metrics['Random Uniform'] = train_evaluate_model(dummy_uniform, X, y)
    
    # # 2. Dummy classifier (most frequent)
    # print("\nEvaluating majority class baseline...")
    # dummy_majority = Pipeline([
    #     ('tfidf', TfidfVectorizer()),  # Dummy transform that will be ignored
    #     ('clf', DummyClassifier(strategy='most_frequent'))
    # ])
    # model_metrics['Majority Class'] = train_evaluate_model(dummy_majority, X, y)
    
    # 3. Perform hyperparameter tuning for TF-IDF + LR
    # print("\nPerforming hyperparameter tuning for TF-IDF + Logistic Regression...")
    # logreg_model = hyperparameter_tuning(X, y, model_type="logreg")
    # model_metrics['Logistic Regression'] = train_evaluate_model(logreg_model, X, y)
    
    # 4. Naive Bayes
    # print("\nPerforming hyperparameter tuning for TF-IDF + Naive Bayes...")
    # nb_model = hyperparameter_tuning(X, y, model_type="nb")
    # model_metrics['Naive Bayes'] = train_evaluate_model(nb_model, X, y)
    
    # 5. Support Vector Machine
    # print("\nPerforming hyperparameter tuning for TF-IDF + SVM...")
    # svm_model = hyperparameter_tuning(X, y, model_type="svm")
    # model_metrics['SVM'] = train_evaluate_model(svm_model, X, y)
    
    # 6. Random Forest
    # print("\nPerforming hyperparameter tuning for TF-IDF + Random Forest...")
    # rf_model = hyperparameter_tuning(X, y, model_type="rf")
    # model_metrics['Random Forest'] = train_evaluate_model(rf_model, X, y)
    
    # 7. Gradient Boosting (this can be slow, so it's optional)
    # print("\nPerforming hyperparameter tuning for TF-IDF + Gradient Boosting...")
    # gb_model = hyperparameter_tuning(X, y, model_type="gb")
    # model_metrics['Gradient Boosting'] = train_evaluate_model(gb_model, X, y)
    
    # 8. K-Nearest Neighbors
    print("\nPerforming hyperparameter tuning for TF-IDF + KNN...")
    knn_model = hyperparameter_tuning(X, y, model_type="knn")
    model_metrics['KNN'] = train_evaluate_model(knn_model, X, y)
    
    # # 9. Include BERT results if available (using provided accuracy)
    # # Note: Ideally, we would run the BERT model with k-fold CV as well
    # bert_acc = 0.858  # Provided BERT accuracy
    # bert_std = 0.02   # Assumed standard deviation (adjust based on actual BERT runs)
    
    # # Include BERT results in the model metrics dictionary
    # model_metrics['BERT'] = {
    #     'Accuracy': (bert_acc, bert_std),
    #     'Precision': (0.85, 0.03),  # Example values, replace with actual
    #     'Recall': (0.86, 0.03),     # Example values, replace with actual
    #     'F1 Score': (0.855, 0.025)  # Example values, replace with actual
    # }
    
    # Create summary table
    print("\nCreating results table...")
    results_df = create_results_table(model_metrics)
    print(results_df)
    
    # Plot metrics comparison
    # print("\nPlotting metrics comparison...")
    # plot_metrics_comparison(model_metrics)
    
    print("\nDone! Results saved to CSV and PNG files.")

if __name__ == "__main__":
    main()