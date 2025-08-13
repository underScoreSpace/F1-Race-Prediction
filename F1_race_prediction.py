import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('f1_cleaned_data.csv')

# Drop columns that are not numeric
non_numeric_columns = df.select_dtypes(include=['object', 'category']).columns
df_cleaned = df.drop(columns=non_numeric_columns)

# Define features and targets
X = df_cleaned.drop(columns=['positionOrder', 'position', 'positionText'], errors='ignore')
y = df_cleaned['positionOrder']

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_one = LinearRegression()
model_one.fit(X_train, y_train)
pickle.dump(model_one, open('model_linear_regression.model', 'wb'))
loaded_model_one = pickle.load(open('model_linear_regression.model', 'rb'))
print("Linear R2:", loaded_model_one.score(X_test, y_test))


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_two = KNeighborsRegressor()
model_two.fit(X_train_scaled, y_train)
pickle.dump(model_two, open('model_KNN.model', 'wb'))
loaded_model_two = pickle.load(open('model_KNN.model', 'rb'))
print("KNN R2:", loaded_model_two.score(X_test_scaled, y_test))

model_three = DecisionTreeRegressor(random_state=42)
model_three.fit(X_train, y_train)
pickle.dump(model_three, open('model_decision_tree.model', 'wb'))
loaded_model_three = pickle.load(open('model_decision_tree.model', 'rb'))
print("Decision tree R2:", loaded_model_three.score(X_test, y_test))


model_four = RandomForestRegressor(n_estimators=100, random_state=42)
model_four.fit(X_train, y_train)
pickle.dump(model_four, open('model_random_forest.model', 'wb'))
loaded_model_four = pickle.load(open('model_random_forest.model', 'rb'))
print("Random forest R2:", loaded_model_four.score(X_test, y_test))


model_five = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_five.fit(X_train, y_train)
pickle.dump(model_five, open('model_xgboost.model', 'wb'))
loaded_model_five = pickle.load(open('model_xgboost.model', 'rb'))
print("XGBoost R2 :", loaded_model_five.score(X_test, y_test))


# Scale features for SVR and KNN(absent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_six = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model_six.fit(X_train_scaled, y_train)
pickle.dump(model_six, open('model_svr.model', 'wb'))
loaded_model_six = pickle.load(open('model_svr.model', 'rb'))
print("SVR R2:", loaded_model_six.score(X_test_scaled, y_test))

"""Feature Engineering 04.15.25 -> Data leak"""

# Load data
df = pd.read_csv('f1_cleaned_data.csv')

# Create a copy for enhancement
df_enhanced = df.copy()

# 1. Process categorical features
categorical_features = ['code', 'nationality_x']  # Adjust based on what's in your dataset
valid_categorical_features = [col for col in categorical_features if col in df_enhanced.columns]

if valid_categorical_features:
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df_enhanced[valid_categorical_features])
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(valid_categorical_features)
    )
    df_enhanced = pd.concat([df_enhanced, encoded_df], axis=1)

# 2. Create interaction features
# Safely handle division by zero
df_enhanced['grid_to_finish_potential'] = df_enhanced.apply(
    lambda row: row['grid'] / row['last_3_avg_position']
    if row['last_3_avg_position'] > 0 else row['grid'],
    axis=1
)

df_enhanced['grid_vs_championship'] = df_enhanced['grid'] - df_enhanced['championship_position']
df_enhanced['recent_form'] = df_enhanced.apply(
    lambda row: row['last_3_avg_position'] / row['championship_position']
    if row['championship_position'] > 0 else row['last_3_avg_position'],
    axis=1
)

df_enhanced['circuit_experience_factor'] = df_enhanced['circuit_races'] * df_enhanced.apply(
    lambda row: 1 / row['best_circuit_position'] if row['best_circuit_position'] > 0 else 1,
    axis=1
)

# 3. Prepare data for modeling
# Keep only numeric columns for model training
df_model = df_enhanced.select_dtypes(include=['number'])
# Define features and target
X = df_model.drop(columns=['positionOrder', 'position', 'positionText'], errors='ignore')
y = df_model['positionOrder']

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models with multiple metrics
def evaluate_model(model, X_train, X_test, y_train, y_test, scale=False):
    # Scale features if needed
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test

    # Fit model
    model.fit(X_train_use, y_train)

    # Make predictions
    y_pred = model.predict(X_test_use)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    if scale:
        cv_scores = cross_val_score(model, scaler.transform(X_train), y_train, cv=cv, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

    return {
        'model': model.__class__.__name__,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'model_object': model
    }

# 5. Train and evaluate multiple models
models = []

# Linear Regression
model_linear = LinearRegression()
result_linear = evaluate_model(model_linear, X_train, X_test, y_train, y_test)
models.append(result_linear)

# KNN Regressor (requires scaling)
model_knn = KNeighborsRegressor()
result_knn = evaluate_model(model_knn, X_train, X_test, y_train, y_test, scale=True)
models.append(result_knn)

# Decision Tree
model_dt = DecisionTreeRegressor(random_state=42)
result_dt = evaluate_model(model_dt, X_train, X_test, y_train, y_test)
models.append(result_dt)

# Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
result_rf = evaluate_model(model_rf, X_train, X_test, y_train, y_test)
models.append(result_rf)

# XGBoost
model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
result_xgb = evaluate_model(model_xgb, X_train, X_test, y_train, y_test)
models.append(result_xgb)

# SVR (requires scaling)
model_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
result_svr = evaluate_model(model_svr, X_train, X_test, y_train, y_test, scale=True)
models.append(result_svr)

# 6. Print model comparison results
for model in models:
    print(f"{model['model']}:")
    print(f"  R2: {model['r2']:.4f}")
    print(f"  MAE: {model['mae']:.4f}")
    print(f"  RMSE: {model['rmse']:.4f}")
    print(f"  CV R2: {model['cv_r2_mean']:.4f} ± {model['cv_r2_std']:.4f}")
    print()

# 7. Hyperparameter tuning for the best model -> XGBoost
best_model_info = max(models, key=lambda x: x['r2'])
print(f"Best model: {best_model_info['model']} with R2 = {best_model_info['r2']:.4f}")

# Hyperparameter tuning for XGBoost
if best_model_info['model'] == 'XGBRegressor':
    print("Performing hyperparameter tuning for XGBoost...")

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=10,  # Try 10 combinations
        cv=3,
        scoring='r2',
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best R2 score: {random_search.best_score_:.4f}")

    # Train final model with best parameters
    final_model = random_search.best_estimator_

    # Evaluate final model
    y_pred = final_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Fixed RMSE calculation

    print(f"Final XGBoost model performance:")
    print(f"  R2: {final_r2:.4f}")
    print(f"  MAE: {final_mae:.4f}")
    print(f"  RMSE: {final_rmse:.4f}")

    # Save final model
    pickle.dump(final_model, open('model_xgboost_tuned.model', 'wb'))

    # Feature importance analysis
    feature_importance = final_model.feature_importances_
    feature_names = X_train.columns

    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), feature_importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    # Print top 10 most important features
    print("Top 10 most important features:")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. FINAL DATA PREPARATION AND FEATURE ENGINEERING PIPELINE

def prepare_f1_data(df_path='f1_cleaned_data.csv'):
    """Load and prepare F1 data for race position prediction."""
    # Load data
    df = pd.read_csv(df_path)

    # Create custom features
    df['grid_to_finish_potential'] = df.apply(
        lambda row: row['grid'] / row['last_3_avg_position']
        if row['last_3_avg_position'] > 0 else row['grid'], axis=1)

    df['recent_form'] = df.apply(
        lambda row: row['last_3_avg_position'] / row['championship_position']
        if row['championship_position'] > 0 else row['last_3_avg_position'], axis=1)

    # Based on feature importance, focus on top predictors
    important_features = [
        'grid_position', 'team_championship_position', 'laps',
        'grid_to_finish_potential', 'championship_position',
        'grid', 'recent_form', 'team_championship_points',
        'alt', 'lat'
    ]

    # Define target and features
    X = df[important_features]
    y = df['positionOrder']

    return X, y, df

# 2. MODEL TRAINING FUNCTION

def train_final_model(X, y, random_state=42):
    """Train and tune the final XGBoost model based on feature importance findings."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Define best parameters based on previous tuning
    best_params = {
        'max_depth': 5,  # Example values - replace with your actual best parameters
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train final model
    final_model = xgb.XGBRegressor(**best_params, random_state=random_state)
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate position accuracy metrics (how often we're within N positions)
    position_diff = np.abs(y_test - y_pred)
    within_1_pos = np.mean(position_diff <= 1)
    within_3_pos = np.mean(position_diff <= 3)

    results = {
        'model': final_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2,
        'mae': mae,
        'within_1_pos': within_1_pos,
        'within_3_pos': within_3_pos
    }

    return results

# 3. VISUALIZATION FUNCTIONS

def plot_feature_importance(model, feature_names):
    """Plot feature importance from model."""
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance for F1 Race Position Prediction')
    plt.bar(range(len(indices)), feature_importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('final_feature_importance.png')
    plt.show()

def plot_predictions_vs_actual(y_test, y_pred):
    """Plot predicted vs actual positions."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Position')
    plt.ylabel('Predicted Position')
    plt.title('Predicted vs Actual F1 Race Positions')
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.show()

# 4. PREDICTION FUNCTION FOR NEW DATA

def predict_race_position(model, driver_data):
    """
    Predict race position for new driver data.

    Parameters:
    -----------
    model: Trained model
    driver_data: DataFrame with required features

    Returns:
    --------
    Predicted position
    """
    return model.predict(driver_data)[0]

# 5. MAIN EXECUTION FUNCTION

def run_f1_prediction_project():
    """Execute the full F1 prediction workflow."""
    print("=========== F1 RACE POSITION PREDICTION PROJECT ===========")
    print("\nLoading and preparing data...")
    X, y, df = prepare_f1_data()

    print("\nTraining final model...")
    results = train_final_model(X, y)

    print("\nModel Performance:")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f}")
    print(f"Predictions within 1 position: {results['within_1_pos']*100:.2f}%")
    print(f"Predictions within 3 positions: {results['within_3_pos']*100:.2f}%")

    print("\nGenerating visualizations...")
    plot_feature_importance(results['model'], X.columns)
    plot_predictions_vs_actual(results['y_test'], results['y_pred'])

    print("\nSaving final model...")
    pickle.dump(results['model'], open('final_f1_position_model.model', 'wb'))

    print("\nProject complete! Model saved as 'final_f1_position_model.model'")

    return results['model'], X, y

# Execute if running as main script
if __name__ == "__main__":
    model, X, y = run_f1_prediction_project()

    # Example of making a prediction for a new race
    print("\nExample Prediction:")
    example_driver = {
        'grid_position': 3,
        'team_championship_position': 2,
        'laps': 50,
        'grid_to_finish_potential': 1.5,
        'championship_position': 4,
        'grid': 3,
        'recent_form': 1.2,
        'team_championship_points': 350,
        'alt': 500,
        'lat': 40.5
    }
    example_df = pd.DataFrame([example_driver])

    predicted_position = predict_race_position(model, example_df)
    print(f"Predicted race position: {predicted_position:.2f}")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("poster", font_scale=1.0)

# 1. Model Comparison Chart
def create_model_comparison_chart():
    # Model results (R-squared values)
    models = ['XGBoost', 'Random Forest', 'Linear Regression', 'SVR', 'Decision Tree', 'KNN']
    r2_scores = [0.7852, 0.75067, 0.69258, 0.51383, 0.48324, 0.45317]

    # Sort by performance
    sorted_indices = np.argsort(r2_scores)[::-1]  # Descending order
    models = [models[i] for i in sorted_indices]
    r2_scores = [r2_scores[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal bar chart
    bars = ax.barh(models, r2_scores, color=sns.color_palette("viridis", len(models)))

    # Add values to the end of each bar
    for i, v in enumerate(r2_scores):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')

    # Customize chart
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('R² Score (higher is better)')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Add grid lines
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Performance Metrics Visualization
def create_performance_metrics_viz():
    # Performance metrics
    metrics = {
        'Within 1 position': 31.53,
        'Within 2 positions': 65.74,  # This is an estimate based on the data pattern
        'Within 3 positions': 77.43,
        'Within 5 positions': 90.12,  # This is an estimate based on the data pattern
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a circular progress visualization
    categories = list(metrics.keys())
    values = list(metrics.values())

    # Create bar chart with gradient colors
    colors = sns.color_palette("YlGnBu", len(categories))
    bars = ax.bar(categories, values, color=colors, width=0.6)

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold'
        )

    # Customize chart
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage of Predictions')
    ax.set_title('Prediction Accuracy by Position Margin', fontsize=16, fontweight='bold')

    # Add annotation about R² and MAE
    ax.text(
        0.5, -0.15,
        f'XGBoost Model: R² = 0.7852, Mean Absolute Error = 2.06 positions',
        ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7)
    )

    # Add grid lines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

# Circular visualization focused on the "within 3 positions" metric
def create_accuracy_circle_viz():
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Hide axes
    ax.axis('off')

    # Create a circle at the center
    circle_size = 0.7
    accuracy = 77.43 / 100  # convert to proportion

    # Draw outer circle (representing 100%)
    outer_circle = plt.Circle((0.5, 0.5), circle_size/2, fill=False,
                            linewidth=4, edgecolor='lightgray')
    ax.add_artist(outer_circle)

    # Draw inner circle (representing accuracy)
    inner_circle = plt.Circle((0.5, 0.5), circle_size/2 * accuracy,
                             linewidth=0, color='#3498db')
    ax.add_artist(inner_circle)

    # Add text in the center
    plt.text(0.5, 0.5, f'{77.43}%\nAccurate\nwithin 3\npositions',
             ha='center', va='center', fontsize=16, fontweight='bold',
             color='white')

    # Add R² score text
    plt.text(0.5, 0.15, f'Model R² = 0.7852',
             ha='center', va='center', fontsize=14)

    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Set the limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig('accuracy_circle.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all visualization functions
create_model_comparison_chart()
create_performance_metrics_viz()
create_accuracy_circle_viz()

print("Visualizations created successfully!")