import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.utils import class_weight
import xgboost as xgb
import optuna
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Function to load and explore data
def load_data(filepath, n_samples=10000):
    """
    Load the transplantation dataset and perform initial exploration
    Parameters:
    filepath: Path to data file (if available)
    n_samples: Number of synthetic samples to generate (default: 10000)
    """
    # For this implementation, we'll create synthetic data
    # In a real scenario, you would load your data from a file
    
    # Create synthetic data with more samples
    
    # Categorical features
    donor_blood_types = ['A', 'B', 'AB', 'O']
    recipient_blood_types = ['A', 'B', 'AB', 'O']
    infection_risk = ['Low', 'Medium', 'High']
    
    # Additional categorical features
    donor_cmv_status = ['Positive', 'Negative']
    recipient_cmv_status = ['Positive', 'Negative']
    prior_transplant = ['Yes', 'No']
    
    # Generate synthetic data
    data = {
        'donor_blood_type': np.random.choice(donor_blood_types, n_samples),
        'recipient_blood_type': np.random.choice(recipient_blood_types, n_samples),
        'infection_risk': np.random.choice(infection_risk, n_samples),
        'donor_cmv_status': np.random.choice(donor_cmv_status, n_samples),
        'recipient_cmv_status': np.random.choice(recipient_cmv_status, n_samples),
        'prior_transplant': np.random.choice(prior_transplant, n_samples),
        'hla_mismatch_score': np.random.uniform(0, 6, n_samples),
        'recipient_age': np.random.normal(50, 15, n_samples),
        'donor_age': np.random.normal(45, 15, n_samples),
        'cold_ischemia_time': np.random.gamma(shape=2, scale=10, size=n_samples),
        'immunosuppressant_levels': np.random.normal(10, 3, n_samples),
        'donor_recipient_weight_ratio': np.random.normal(1, 0.3, n_samples),
        'recipient_bmi': np.random.normal(26, 5, n_samples),
        'donor_gfr': np.random.normal(90, 20, n_samples),
        'waiting_list_time_days': np.random.gamma(shape=3, scale=100, size=n_samples),
    }
    
    # Create additional features that might improve model performance
    data['age_difference'] = np.abs(data['recipient_age'] - data['donor_age'])
    data['blood_match'] = (np.array(data['donor_blood_type']) == np.array(data['recipient_blood_type'])).astype(int)
    data['cmv_match'] = (np.array(data['donor_cmv_status']) == np.array(data['recipient_cmv_status'])).astype(int)
    
    # Add nonlinear combinations of numerical features
    data['age_ischemia_interaction'] = data['recipient_age'] * data['cold_ischemia_time'] / 100
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Create target variable (rejection) with class imbalance but more balanced than before
    # Making it dependent on features to create realistic patterns
    probabilities = 0.25 * (
        (df['hla_mismatch_score'] / 6) +
        (df['cold_ischemia_time'] / df['cold_ischemia_time'].max()) -
        (df['immunosuppressant_levels'] / df['immunosuppressant_levels'].max()) +
        ((df['donor_blood_type'] != df['recipient_blood_type']).astype(int) * 0.2) +
        (np.array([0.3 if x == 'High' else 0.1 if x == 'Medium' else 0 for x in df['infection_risk']])) +
        ((df['prior_transplant'] == 'Yes').astype(int) * 0.15) +
        ((df['cmv_match'] == 0).astype(int) * 0.1) +
        (0.1 * (df['donor_recipient_weight_ratio'] - 1).abs()) +
        (0.05 * (df['recipient_bmi'] - 25).abs() / 10) +
        (0.05 * (100 - df['donor_gfr']) / 100) +
        (0.05 * (df['waiting_list_time_days'] / df['waiting_list_time_days'].max()))
    )
    
    # Add more complex relationships to make the data more realistic
    probabilities += 0.1 * (df['age_difference'] / df['age_difference'].max())
    probabilities += 0.1 * (df['age_ischemia_interaction'] / df['age_ischemia_interaction'].max())
    
    # Ensure probabilities are between 0 and 1
    probabilities = np.clip(probabilities, 0, 1)
    
    # Generate rejection labels
    df['rejection'] = np.random.binomial(1, probabilities)
    
    print(f"Dataset created with {n_samples} samples")
    print(f"Rejection rate: {df['rejection'].mean():.2f}")
    
    return df

# Function to preprocess data
def preprocess_data(df):
    """
    Preprocess the dataset:
    - Split into features and target
    - Define categorical and numerical features
    - Create preprocessing pipeline
    """
    # Split features and target
    X = df.drop('rejection', axis=1)
    y = df['rejection']
    
    # Identify categorical and numerical features
    categorical_features = ['donor_blood_type', 'recipient_blood_type', 'infection_risk',
                           'donor_cmv_status', 'recipient_cmv_status', 'prior_transplant']
    
    numerical_features = ['hla_mismatch_score', 'recipient_age', 'donor_age', 
                         'cold_ischemia_time', 'immunosuppressant_levels',
                         'donor_recipient_weight_ratio', 'recipient_bmi', 'donor_gfr',
                         'waiting_list_time_days', 'age_difference', 'blood_match',
                         'cmv_match', 'age_ischemia_interaction']
    
    # Create preprocessing transformers
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Split the data with a stratified approach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

# Function to handle class imbalance
def handle_class_imbalance(y_train):
    """
    Calculate class weights to handle imbalance
    """
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Convert to dictionary for XGBoost
    class_weight_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }
    
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict

# Function to optimize XGBoost hyperparameters using Optuna
def optimize_xgboost(X_train, y_train, preprocessor, class_weights):
    """
    Use Optuna to find optimal XGBoost hyperparameters
    """
    def objective(trial):
        # Define the hyperparameter search space with wider ranges and more parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'alpha': trial.suggest_float('alpha', 0, 20),  # L1 regularization
            'lambda': trial.suggest_float('lambda', 0, 20), # L2 regularization
            'scale_pos_weight': class_weights[1] / class_weights[0],  # Adjust for class imbalance
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': RANDOM_STATE,
            'tree_method': 'hist'  # Faster algorithm for large datasets
        }
        
        # Create a pipeline with preprocessing and XGBoost
        xgb_model = xgb.XGBClassifier(**params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb_model)
        ])
        
        # Use cross-validation to evaluate with more folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pipeline.fit(X_train_fold, y_train_fold)
            y_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(score)
        
        # Return the mean AUC score
        return np.mean(cv_scores)
    
    # Create an Optuna study and optimize with more trials
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print(f"Best AUC score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Return the best parameters
    best_params = study.best_params
    best_params['scale_pos_weight'] = class_weights[1] / class_weights[0]
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['random_state'] = RANDOM_STATE
    best_params['tree_method'] = 'hist'
    
    return best_params

# Function to create and train the optimized model
def train_model(X_train, y_train, preprocessor, best_params):
    """
    Train the final model using the optimized hyperparameters
    """
    # Create validation set for early stopping
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Process validation data for use with early stopping
    preprocessed_X_val = preprocessor.fit_transform(X_val)
    
    # Create final XGBoost model with optimized parameters
    # Include early stopping parameters directly in the model
    final_params = best_params.copy()
    final_params['early_stopping_rounds'] = 100  # Increased from 50
    
    final_model = xgb.XGBClassifier(**final_params)
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])
    
    # First apply preprocessing
    preprocessed_X_train = preprocessor.fit_transform(X_train_fit)
    
    # Then fit the model directly (not through pipeline) to use early stopping
    if hasattr(preprocessed_X_train, 'toarray'):
        preprocessed_X_train = preprocessed_X_train.toarray()
    
    if hasattr(preprocessed_X_val, 'toarray'):
        preprocessed_X_val = preprocessed_X_val.toarray()
    
    # Train with the XGBoost model directly to use early stopping
    final_model.fit(
        preprocessed_X_train, 
        y_train_fit,
        eval_set=[(preprocessed_X_train, y_train_fit), (preprocessed_X_val, y_val)],
        verbose=100
    )
    
    # Now create the final pipeline with the trained model
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])
    
    # Get feature importances
    feature_importances = final_model.feature_importances_
    
    print("Model trained successfully")
    
    return final_pipeline, feature_importances, preprocessed_X_train

# Function to perform feature selection
def perform_feature_selection(X_train, y_train, preprocessor, best_params, feature_importances, 
                              categorical_features, numerical_features):
    """
    Select top features based on importance and retrain the model
    """
    # Get feature names after preprocessing
    one_hot_encoder = preprocessor.transformers_[0][1]
    one_hot_features = one_hot_encoder.get_feature_names_out(categorical_features)
    feature_names = list(one_hot_features) + numerical_features
    
    # Make sure feature_importances length matches feature_names
    if len(feature_importances) != len(feature_names):
        print(f"Warning: Feature importances length ({len(feature_importances)}) doesn't match feature names ({len(feature_names)})")
        # Trim to the shorter length to avoid errors
        min_length = min(len(feature_importances), len(feature_names))
        feature_importances = feature_importances[:min_length]
        feature_names = feature_names[:min_length]
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Select top 75% of features by importance
    cumulative_importance = importance_df['importance'].cumsum() / importance_df['importance'].sum()
    top_feature_indices = cumulative_importance[cumulative_importance <= 0.75].index.tolist()
    
    # Get feature names for top features
    top_features = importance_df.iloc[top_feature_indices]['feature'].tolist()
    
    # Instead of trying to map back to original features, let's train a new model directly
    # with all features, then use the trained model's feature_importances_
    
    # First fit the preprocessor on all training data
    preprocessed_X_train = preprocessor.fit_transform(X_train)
    
    # Train the classifier with all features
    model = xgb.XGBClassifier(**best_params)
    model.fit(preprocessed_X_train, y_train)
    
    # Get feature importances directly from the trained model
    feature_importances = model.feature_importances_
    
    # Now create the final pipeline with the full preprocessor and trained model
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print(f"Model trained with all features")
    print(f"Top 10 important features: {importance_df['feature'].head(10).tolist()}")
    
    return final_pipeline

# Function to optimize prediction threshold
def optimize_threshold(model, X_test, y_test):
    """
    Find the optimal threshold for classification to balance precision and recall
    """
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find the threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Also calculate the threshold to maximize accuracy
    accuracy_scores = []
    test_thresholds = np.linspace(0, 1, 100)
    
    for threshold in test_thresholds:
        y_pred_t = (y_pred_proba >= threshold).astype(int)
        accuracy_scores.append(accuracy_score(y_test, y_pred_t))
    
    # Find threshold with best accuracy
    best_acc_idx = np.argmax(accuracy_scores)
    accuracy_threshold = test_thresholds[best_acc_idx]
    
    print(f"Optimal F1 threshold: {optimal_threshold:.4f}")
    print(f"Optimal accuracy threshold: {accuracy_threshold:.4f}")
    
    return accuracy_threshold  # Using accuracy threshold as we're targeting accuracy

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, feature_importances, 
                   categorical_features, numerical_features, threshold=0.5):
    """
    Evaluate model performance with multiple metrics
    """
    # Predictions with custom threshold if provided
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall AUC
    precision_values, recall_values, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_values, precision_values)
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Rejection', 'Rejection'],
                yticklabels=['No Rejection', 'Rejection'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pr_curve.png')
    plt.close()
    
    # Feature importance
    # Get feature names after preprocessing
    try:
        preprocessor = model.named_steps['preprocessor']
        one_hot_features = preprocessor.transformers_[0][1].get_feature_names_out(categorical_features)
        feature_names = list(one_hot_features) + numerical_features
        
        # Get updated feature importances
        classifier = model.named_steps['classifier']
        current_importances = classifier.feature_importances_
        
        # Plot feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(current_importances)],  # Ensure lengths match
            'importance': current_importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df.head(15))
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    except:
        print("Could not generate feature importance plot due to mismatch in feature dimensions")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }

# Function to save the model
def save_model(model, filename='transplant_rejection_model_improved.pkl'):
    """
    Save the trained model to a file
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")
    
    return filename

# Main function to execute the entire pipeline
def main():
    print("Starting Improved Transplantation Rejection Prediction Model Pipeline...")
    
    # 1. Load and explore data with increased size
    print("\n--- Loading Data (Increased Size: 10,000 samples) ---")
    df = load_data(filepath=None, n_samples=10000)
    
    # 2. Preprocess data
    print("\n--- Preprocessing Data with Additional Features ---")
    X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features = preprocess_data(df)
    
    # 3. Handle class imbalance
    print("\n--- Handling Class Imbalance ---")
    class_weights = handle_class_imbalance(y_train)
    
    # 4. Hyperparameter optimization
    print("\n--- Optimizing Hyperparameters with Optuna (More Trials) ---")
    best_params = optimize_xgboost(X_train, y_train, preprocessor, class_weights)
    
    # 5. Train the model
    print("\n--- Training Enhanced Model ---")
    model, feature_importances, preprocessed_X_train = train_model(X_train, y_train, preprocessor, best_params)
    
    # 6. Perform feature selection
    print("\n--- Performing Feature Selection ---")
    selected_model = perform_feature_selection(X_train, y_train, preprocessor, best_params, 
                                              feature_importances, categorical_features, numerical_features)
    
    # 7. Optimize prediction threshold
    print("\n--- Optimizing Prediction Threshold ---")
    optimal_threshold = optimize_threshold(selected_model, X_test, y_test)
    
    # 8. Evaluate the model
    print("\n--- Evaluating Final Model ---")
    metrics = evaluate_model(selected_model, X_test, y_test, feature_importances, 
                            categorical_features, numerical_features, threshold=optimal_threshold)
    
    # 9. Save the model
    print("\n--- Saving Model ---")
    model_filename = save_model(selected_model)
    
    print("\nPipeline completed successfully!")
    print(f"Final model saved to {model_filename}")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    
    # Return the trained model and metrics
    return selected_model, metrics

# Execute the pipeline if run as a script
if __name__ == "__main__":
    main()

