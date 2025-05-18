import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

def load_data(data_dir="pyScrcClient/data"):
    """
    Load all CSV files from the data directory and its subdirectories.
    
    Args:
        data_dir: Directory containing collected TORCS data
        
    Returns:
        Combined DataFrame of all CSV files
    """
    # Find all CSV files in the data directory and subdirectories
    csv_files = []
    for track_dir in os.listdir(data_dir):
        track_path = os.path.join(data_dir, track_dir)
        if os.path.isdir(track_path):
            for csv_file in glob.glob(os.path.join(track_path, "*.csv")):
                csv_files.append(csv_file)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add metadata: track and car info from filename
            parts = os.path.basename(file).split('_')
            if len(parts) >= 3:
                df['track'] = parts[0]
                df['car'] = parts[1]
                df['mode'] = parts[2]
            dataframes.append(df)
            print(f"Loaded {file} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dataframes:
        raise ValueError("No data files could be loaded")
    
    return pd.concat(dataframes, ignore_index=True)

def preprocess_data(df, test_size=0.2):
    """
    Preprocess the data for training.
    
    Args:
        df: DataFrame with TORCS data
        test_size: Fraction of data to use for validation
        
    Returns:
        X_train, Y_train, X_val, Y_val: Training and validation data
    """
    # Remove unnecessary columns for training
    cols_to_drop = []
    
    # Detect columns that might not be useful for training
    for col in df.columns:
        # Remove timestamp, run ID, frame, etc.
        if col in ['Timestamp', 'RunID', 'Frame']:
            cols_to_drop.append(col)
        # Remove constant or nearly constant columns
        if df[col].nunique() < 3:
            cols_to_drop.append(col)
    
    # Keep track and car information for analysis if present
    metadata_cols = ['track', 'car', 'mode']
    for col in metadata_cols:
        if col in df.columns:
            cols_to_drop.append(col)
    
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Define input and output columns
    # Input: Everything except control outputs
    # Output: Steering, acceleration, brake, clutch, gear
    output_cols = ['Acceleration', 'Braking', 'Clutch', 'Gear', 'Steering']
    
    # Check if output columns exist in the dataset
    for col in output_cols:
        if col not in df_clean.columns:
            raise ValueError(f"Output column {col} not found in data")
    
    input_cols = [col for col in df_clean.columns if col not in output_cols]
    
    # Convert all columns to float to avoid 'object' dtype issues
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Replace any NaN values with 0
    df_clean = df_clean.fillna(0)
    
    # Split into training and validation sets
    n_samples = len(df_clean)
    train_indices = np.random.choice(np.arange(n_samples), int(n_samples * (1 - test_size)), replace=False)
    val_indices = list(set(np.arange(n_samples)) - set(train_indices))
    
    train_data = df_clean.iloc[train_indices].reset_index(drop=True)
    val_data = df_clean.iloc[val_indices].reset_index(drop=True)
    
    # Normalize input data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[input_cols])
    X_val = scaler.transform(val_data[input_cols])
    
    Y_train = train_data[output_cols].values.astype('float32')
    Y_val = val_data[output_cols].values.astype('float32')
    
    print(f"Input features: {len(input_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Convert lists to numpy arrays
    input_cols = np.array(input_cols)
    output_cols = np.array(output_cols)
    
    return X_train, Y_train, X_val, Y_val, scaler, input_cols, output_cols

def create_model(input_dim, output_dim):
    """
    Create a neural network model for TORCS control.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output controls
        
    Returns:
        Compiled Keras model
    """
    learning_rate = 3e-3
    
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(output_dim, kernel_initializer='he_normal'))
    
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error', metrics=['mae'])
    
    return model

def train_model(X_train, Y_train, X_val, Y_val, batch_size=256, epochs=100):
    """
    Train the neural network model.
    
    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        Trained model and training history
    """
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    # Use the fixed create_model function
    model = create_model(input_dim, output_dim)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    return model, history

def plot_training_history(history, model_dir='model'):
    """Plot the training and validation loss."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot log-scale loss
    plt.subplot(2, 2, 3)
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('Model Loss (log scale)')
    plt.ylabel('Log Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot learning progress as percent improvement
    plt.subplot(2, 2, 4)
    initial_loss = history.history['loss'][0]
    loss_percent = [(loss/initial_loss)*100 for loss in history.history['loss']]
    val_initial_loss = history.history['val_loss'][0]
    val_loss_percent = [(loss/val_initial_loss)*100 for loss in history.history['val_loss']]
    plt.plot(loss_percent)
    plt.plot(val_loss_percent)
    plt.title('Learning Progress')
    plt.ylabel('% of Initial Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot to the model directory
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.savefig(os.path.join(model_dir, 'training_history.pdf'))
    
    # Generate additional plot for steering performance
    plt.figure(figsize=(15, 5))
    plt.plot(history.history['loss'])
    plt.title('Training Progress')
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(model_dir, 'training_progress.png'))
    plt.savefig(os.path.join(model_dir, 'training_progress.pdf'))
    
    plt.close('all')  # Close all figures to prevent display

def save_model(model, scaler, input_cols, output_cols, model_dir='model'):
    """
    Save the model and preprocessing information.
    
    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        input_cols: List of input column names
        output_cols: List of output column names
        model_dir: Directory for saving model files
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "torcs_model.h5")
    model.save(model_path)
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, "torcs_model_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save column information
    input_cols_path = os.path.join(model_dir, "torcs_model_input_cols.npy")
    output_cols_path = os.path.join(model_dir, "torcs_model_output_cols.npy")
    np.save(input_cols_path, input_cols)
    np.save(output_cols_path, output_cols)
    
    # Generate a model summary file
    summary_path = os.path.join(model_dir, "model_summary.txt")
    with open(summary_path, 'w') as f:
        # Redirect model summary to the file
        from contextlib import redirect_stdout
        with redirect_stdout(f):
            model.summary()
        
        # Add additional information
        f.write("\n\nModel Training Information:\n")
        f.write(f"Input features: {len(input_cols)}\n")
        f.write(f"Output features: {len(output_cols)}\n")
        f.write(f"Model saved at: {model_path}\n")
    
    print(f"Model and related files saved to {model_dir}/")
    return model_path

def main():
    """Main function to train and save the model."""
    # Create model directory
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Preprocess data
    X_train, Y_train, X_val, Y_val, scaler, input_cols, output_cols = preprocess_data(data)
    
    # Train model
    model, history = train_model(X_train, Y_train, X_val, Y_val, epochs=100)
    
    # Plot training history
    plot_training_history(history, model_dir)
    
    # Save model
    model_path = save_model(model, scaler, input_cols, output_cols, model_dir)
    
    # Test prediction
    sample_input = X_val[0:1]
    prediction = model.predict(sample_input)
    
    print("Sample prediction:")
    for i, col in enumerate(output_cols):
        print(f"{col}: {prediction[0][i]:.4f}")
    
    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Training visualization saved to {model_dir}/training_history.png")

if __name__ == "__main__":
    main() 