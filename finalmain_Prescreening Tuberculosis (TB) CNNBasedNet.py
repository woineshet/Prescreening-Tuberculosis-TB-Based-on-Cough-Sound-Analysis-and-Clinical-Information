#!/usr/bin/env python
# coding: utf-8

# # **Prescreening Tuberculosis (TB) Based on Cough Sound Analysis and Clinical Information Using Machine Learning** 

# ## **import libraries **
# 

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape
import keras_tuner as kt
import shap
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Activation, RepeatVector, Permute
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Average, Concatenate, GRU, Bidirectional, 
    LeakyReLU, Dense, Dropout, Input, 
    Conv1D, Layer, Add, BatchNormalization, 
    MaxPooling1D, GlobalAveragePooling1D, LSTM,
    Multiply, Flatten  
)
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler, 
                                        EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras import regularizers, initializers, constraints
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, average_precision_score)
from tensorflow.keras.losses import (categorical_crossentropy, mean_squared_error, 
                                     binary_crossentropy)
from tensorflow.keras.utils import plot_model
import tf2onnx
from tensorflow.keras.initializers import GlorotUniform


# In[2]:


#2 import libraries
import pandas as pd
import os
import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import onnxruntime as rt
import librosa
from scipy import stats
from scipy import stats
import zipfile
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import traceback
import time
import numpy as np
import os
import numpy as np


# In[3]:


# Configuring the random seed for reproducibility
random_seed = 34
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Specify GPU usage by setting the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

# Define the number of epochs for model training
epochs = 50

# Specify the data folder path, adjust as necessary for your project structure
data_folder_path = ''
import shap  # SHAP for explaining predictions


# ## **visualize_data_distributions**

# In[4]:


def visualize_data_distributions(X_train_label, X_val_label, feature_columns):
    """Create comprehensive visualizations of feature distributions"""
    n_features = len(feature_columns)
    fig, axes = plt.subplots(2, n_features, figsize=(20, 8))
    
    for i, feature in enumerate(feature_columns):
        sns.histplot(data=X_train_label[:, i], ax=axes[0, i], kde=True)
        axes[0, i].set_title(f'{feature}\nDistribution (Training)')
        
        sns.boxplot(data=[X_train_label[:, i], X_val_label[:, i]], ax=axes[1, i])
        axes[1, i].set_title(f'{feature}\nTrain vs. Validation Box Plot')
    
    plt.tight_layout()
    plt.show()


def visualize_feature_relationships(X_train_label, y_train, feature_columns):
    """Create correlation matrix and pair plots (on training data only)"""
    data = pd.DataFrame(X_train_label, columns=feature_columns)
    data['Target'] = y_train

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    correlations = data.corr()
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix (Training Set)')
    plt.tight_layout()
    plt.show()
    
    # Pair Plot
    sns.pairplot(data, hue='Target', diag_kind='kde')
    plt.suptitle('Feature Pair Relationships (Training Set)', y=1.02)
    plt.show()


def print_data_statistics(X_train_label, X_val_label, feature_columns):
    """Print comprehensive statistics about the training and validation data"""
    print("\nData Statistics:")
    print("-" * 50)
    
    for i, feature in enumerate(feature_columns):
        print(f"\n{feature}:")
        print("Training Data:")
        print(f"  Mean: {np.mean(X_train_label[:, i]):.3f}")
        print(f"  Std:  {np.std(X_train_label[:, i]):.3f}")
        print(f"  Min:  {np.min(X_train_label[:, i]):.3f}")
        print(f"  Max:  {np.max(X_train_label[:, i]):.3f}")
        
        print("Validation Data:")
        print(f"  Mean: {np.mean(X_val_label[:, i]):.3f}")
        print(f"  Std:  {np.std(X_val_label[:, i]):.3f}")
        print(f"  Min: {np.min(X_val_label[:, i]):.3f}")
        print(f"  Max: {np.max(X_val_label[:, i]):.3f}")


def save_visualization_results(X_train_label, X_val_label, feature_columns, output_dir='./results'):
    """Save all visualization results comparing train and validation sets"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature distribution plots
    fig_dist = create_feature_plots(X_train_label, X_val_label, feature_columns)
    fig_dist.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    
    # Save correlation matrix (combined train + val)
    df_combined = pd.DataFrame(np.vstack([X_train_label, X_val_label]), columns=feature_columns)
    fig_corr = plot_correlation_matrix(df_combined, feature_columns)
    fig_corr.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    
    # Save preprocessing comparison if raw is attached (optional)
    if hasattr(save_visualization_results, 'X_raw'):
        fig_prep = plot_preprocessing_comparison(
            save_visualization_results.X_raw,
            X_train_label,
            feature_columns
        )
        fig_prep.savefig(os.path.join(output_dir, 'preprocessing_comparison.png'))
    
    plt.close('all')
def visualize_preprocessing_effect(X_raw, X_processed, feature_columns, n_samples=None):
    """Visualize preprocessing effects for each feature"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if n_samples is None:
        n_samples = min(len(X_raw), len(X_processed))
    else:
        n_samples = min(n_samples, len(X_raw), len(X_processed))

    idx = np.random.choice(n_samples, size=n_samples, replace=False)

    X_raw_sample = X_raw[idx]
    X_processed_sample = X_processed[idx]

    n_features = len(feature_columns)
    fig, axes = plt.subplots(2, n_features, figsize=(20, 8))

    for i, feature in enumerate(feature_columns):
        sns.histplot(X_raw_sample[:, i], ax=axes[0, i], kde=True, color='red', alpha=0.5)
        axes[0, i].set_title(f'{feature}\nBefore Preprocessing')

        sns.histplot(X_processed_sample[:, i], ax=axes[1, i], kde=True, color='blue', alpha=0.5)
        axes[1, i].set_title(f'{feature}\nAfter Preprocessing')

    plt.tight_layout()
    plt.show()


# In[12]:


def visualize_raw_distributions_separate(train_raw, val_raw, selected_features, feature_columns):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    selected_indices = [feature_columns.index(f) for f in selected_features]
    train_selected = train_raw[:, selected_indices]
    val_selected = val_raw[:, selected_indices]

    n_features = len(selected_features)
    fig, axes = plt.subplots(3, n_features, figsize=(5 * n_features, 12))

    for i, feature in enumerate(selected_features):
        # Histogram - Train
        sns.histplot(train_selected[:, i], ax=axes[0, i], kde=True, color='skyblue', stat='count', bins=30)
        axes[0, i].set_title(f'{feature} (Train - Raw)')
        axes[0, i].set_xlabel(feature)
        axes[0, i].set_ylabel("Count")

        # Histogram - Val
        sns.histplot(val_selected[:, i], ax=axes[1, i],kde=True, color='salmon', stat='count', bins=30)
        axes[1, i].set_title(f'{feature} (Validation - Raw)')
        axes[1, i].set_xlabel(feature)
        axes[1, i].set_ylabel("Count")

        # Boxplot
        sns.boxplot(data=[train_selected[:, i], val_selected[:, i]], ax=axes[2, i])
        axes[2, i].set_xticklabels(['Train', 'Validation'])
        axes[2, i].set_title(f'{feature} Boxplot (Raw)')

    fig.suptitle("Feature Distributions Before Scaling", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


    
def visualize_scaled_distributions_separate(X_train_label, X_val_label, selected_features, feature_columns):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    selected_indices = [feature_columns.index(f) for f in selected_features]
    train_scaled = X_train_label[:, selected_indices]
    val_scaled = X_val_label[:, selected_indices]

    n_features = len(selected_features)
    fig, axes = plt.subplots(3, n_features, figsize=(5 * n_features, 12))

    for i, feature in enumerate(selected_features):
        # Histogram - Train
        sns.histplot(train_scaled[:, i], ax=axes[0, i], kde=True, color='skyblue', stat='count', bins=30)
        axes[0, i].set_title(f'{feature} (Train - Scaled)')
        axes[0, i].set_xlabel(feature)
        axes[0, i].set_ylabel("Count")

        # Histogram - Val
        sns.histplot(val_scaled[:, i], ax=axes[1, i],kde=True,color='salmon', stat='count', bins=30)
        axes[1, i].set_title(f'{feature} (Validation - Scaled)')
        axes[1, i].set_xlabel(feature)
        axes[1, i].set_ylabel("Count")

        # Boxplot
        sns.boxplot(data=[train_scaled[:, i], val_scaled[:, i]], ax=axes[2, i])
        axes[2, i].set_xticklabels(['Train', 'Validation'])
        axes[2, i].set_title(f'{feature} Boxplot (Scaled)')

    fig.suptitle("Feature Distributions After Scaling", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


#  ## **Model Architecture def function**   

# In[5]:


def CNNbasedNet121(length=22050, seed=42):
    """
    Create base CNNbasedNet121 model with the architecture that achieved 90% accuracy
    """
    input_shape = (length, 1)
    inputs = Input(shape=input_shape)

    # First CNN block
    x = Conv1D(32, kernel_size=64, strides=2, padding='same',
               kernel_initializer=GlorotUniform(seed=seed))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = Dropout(0.2)(x)

    # Second CNN block
    x = Conv1D(64, kernel_size=32, strides=2, padding='same',
               kernel_initializer=GlorotUniform(seed=seed+1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = Dropout(0.2)(x)

    # Third CNN block
    x = Conv1D(128, kernel_size=16, strides=2, padding='same',
               kernel_initializer=GlorotUniform(seed=seed+2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = Dropout(0.2)(x)

    # Global pooling
    x = GlobalAveragePooling1D()(x)

    # Dense layers
    x = Dense(256, kernel_initializer=GlorotUniform(seed=seed+3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid',
                    kernel_initializer=GlorotUniform(seed=seed+4))(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile with the successful configuration
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC()]
    )

    return model


def CNNbasedNet(length, pre_model, seed=42):
    """
    Create enhanced CNNbasedNet with clinical features
    """
    # Audio input branch
    audio_input = Input(shape=(length, 1))
    clinical_input = Input(shape=(16,))

    # Audio processing (using pre-trained model)
    x = audio_input
    for layer in pre_model.layers[1:-2]:
        x = layer(x)

    # Clinical features processing
    y = Dense(64, kernel_initializer=GlorotUniform(seed=seed))(clinical_input)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dropout(0.3)(y)

    y = Dense(32, kernel_initializer=GlorotUniform(seed=seed+1))(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.2)(y)

    # Combine features
    combined = Concatenate()([x, y])

    # Final processing
    z = Dense(128, kernel_initializer=GlorotUniform(seed=seed+2))(combined)
    z = BatchNormalization()(z)
    z = LeakyReLU(alpha=0.2)(z)
    z = Dropout(0.5)(z)

    output = Dense(1, activation='sigmoid',
                   kernel_initializer=GlorotUniform(seed=seed+3))(z)

    model = Model(inputs=[audio_input, clinical_input], outputs=output)

    # Compile with the successful configuration
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC()]
    )

    return model


def train_cnnbasedNet(X_train, X_train_label, y_train, X_val, X_val_label, y_val):
    """
    Train CNNbasedNet with the successful configuration using proper train/val split
    """
    # Create and train base model
    print("Creating and training base model...")
    base_model = CNNbasedNet121(length=22050, seed=42)

    base_history = base_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ],
        verbose=1
    )

    # ✅ Evaluate training performance of base model
    base_train_results = base_model.evaluate(X_train, y_train, verbose=0)
    print(f"\nBase Model Training Accuracy: {base_train_results[1]:.4f}")
    print(f"Base Model Training AUC: {base_train_results[2]:.4f}")

    # Create enhanced model
    print("\nCreating enhanced model...")
    model = CNNbasedNet(length=22050, pre_model=base_model, seed=42)

    # Calculate class weights
    class_weights = dict(enumerate(compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )))

    # Train enhanced model using validation set
    print("\nTraining enhanced model...")
    history = model.fit(
        [X_train, X_train_label],
        y_train,
        validation_data=([X_val, X_val_label], y_val),
        epochs=50,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ],
        verbose=1
    )

    # ✅ Evaluate training performance of enhanced model
    enhanced_train_results = model.evaluate([X_train, X_train_label], y_train, verbose=0)
    print(f"\nEnhanced Model Training Accuracy: {enhanced_train_results[1]:.4f}")
    print(f"Enhanced Model Training AUC: {enhanced_train_results[2]:.4f}")

    return model, history


# ## **Data Loading and Preprocessing Functions**
# 
# 
# 

# In[6]:


def load_and_preprocess_data():
    """
    Load and preprocess the audio and clinical data including additional longitudinal data.
    Updated to ensure no data leakage by splitting before fitting scalers/encoders.
    """
    print("Loading and preprocessing data...")

    try:
        # === Load metadata ===
        clinical_data = pd.read_csv('CODA_TB_Clinical_Meta_Info.csv')
        solicited_data = pd.read_csv('CODA_TB_Solicited_Meta_Info.csv')
        longitudinal_data = pd.read_csv('CODA_TB_Longitudnal_Meta_Info.csv')

        # Merge clinical info with solicited and longitudinal
        merged_solicited = pd.merge(solicited_data, clinical_data, on="participant")
        merged_longitudinal = pd.merge(longitudinal_data, clinical_data, on="participant")
        combined_data = pd.concat([merged_solicited, merged_longitudinal], ignore_index=True)

        # === Feature columns (from CSV) ===
        feature_columns = [
            'sex', 'age', 'height', 'weight', 'reported_cough_dur',
            'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 'tb_prior_Unknown',
            'hemoptysis', 'heart_rate', 'temperature', 'weight_loss',
            'smoke_lweek', 'fever', 'night_sweats'
        ]

        # === Split first (no leakage) ===
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        labels = combined_data['tb_status'].values
        indices = np.arange(len(combined_data))

        train_idx, test_idx = train_test_split(indices, test_size=0.10, stratify=labels, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.20, stratify=labels[train_idx], random_state=42)

        train_df = combined_data.iloc[train_idx].copy()
        val_df = combined_data.iloc[val_idx].copy()
        test_df = combined_data.iloc[test_idx].copy()

        # === Encode categorical features (fit on train only) ===
        categorical_columns = train_df[feature_columns].select_dtypes(include=['object', 'category']).columns
        encoders = {}

        for col in categorical_columns:
            mode = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode)
            val_df[col] = val_df[col].fillna(mode)
            test_df[col] = test_df[col].fillna(mode)

            # === Save raw clinical features before scaling ===
            train_raw = train_df[feature_columns].copy()
            val_raw = val_df[feature_columns].copy()

            encoder = LabelEncoder()
            train_df[col] = encoder.fit_transform(train_df[col].astype(str))
            val_df[col] = encoder.transform(val_df[col].astype(str))
            test_df[col] = encoder.transform(test_df[col].astype(str))
            encoders[col] = encoder

        # === Fill missing numeric values (use train mean only) ===
        for col in feature_columns:
            if col not in categorical_columns:
                mean = train_df[col].mean()
                train_df[col] = train_df[col].fillna(mean)
                val_df[col] = val_df[col].fillna(mean)
                test_df[col] = test_df[col].fillna(mean)

        # === Normalize (fit scaler only on train) ===
        scaler = StandardScaler()
        X_train_label = scaler.fit_transform(train_df[feature_columns])
        X_val_label = scaler.transform(val_df[feature_columns])
        X_test_label = scaler.transform(test_df[feature_columns])

        # === Load audio files from ZIPs ===
        import zipfile, soundfile as sf, librosa

        zip_files = [
            zipfile.ZipFile('solicated_coughs.zip', 'r'),
            zipfile.ZipFile('longitudinal_1.zip', 'r'),
            zipfile.ZipFile('longitudinal_2.zip', 'r')
        ]
        zip_file_map = {name: zipf for zipf in zip_files for name in zipf.namelist()}

        def load_audio(filenames):
            X_audio = []
            for file_name in filenames:
                try:
                    file_name = file_name.strip()
                    if file_name in zip_file_map:
                        with zip_file_map[file_name].open(file_name) as file:
                            audio, rate = sf.read(file)
                            audio = librosa.util.fix_length(audio, size=22050)
                            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-10)
                            X_audio.append(audio.reshape(-1, 1))
                except Exception as e:
                    print(f"Error processing file {file_name}: {str(e)}")
            return np.array(X_audio, dtype=np.float32)

        print(f"Processing {len(combined_data)} audio files...")

        X_train = load_audio(train_df['filename'])
        X_val = load_audio(val_df['filename'])
        X_test = load_audio(test_df['filename'])

        y_train = train_df['tb_status'].values
        y_val = val_df['tb_status'].values
        y_test = test_df['tb_status'].values

        # === Return all as expected ===
        return (
            X_train, X_val, X_test,
            X_train_label.astype(np.float32),
            X_val_label.astype(np.float32),
            X_test_label.astype(np.float32),
            y_train, y_val, y_test,
            train_df, val_df, test_df, feature_columns,  # here
            train_raw.values, val_raw.values    ##added these 4 value for the purpose of data visualization
        )

    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        raise e


# In[7]:


from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encoder_dict = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))  # ensure all values are strings
        encoder_dict[col] = encoder  # store encoder for future use if needed
        
    return df, encoder_dict


# In[ ]:





# In[10]:


###main script, updated the unpacking 

(
    X_train, X_val, X_test,
    X_train_label, X_val_label, X_test_label,
    y_train, y_val, y_test,
    train_df, val_df, test_df, feature_columns,
    train_raw, val_raw
) = load_and_preprocess_data()


# In[ ]:





# In[9]:


# ✅ Call your updated data loading function
X_train, X_val, X_test, X_train_label, X_val_label, X_test_label, y_train, y_val, y_test = load_and_preprocess_data()

# ✅ Print dtype and memory usage of training audio data
print("\n🧠 Audio data (X_train):")
print(" - dtype:", X_train.dtype)
print(" - shape:", X_train.shape)
print(" - memory used:", round(X_train.nbytes / 1024**2, 2), "MB")

# ✅ Print dtype and memory usage of clinical data
print("\n📋 Clinical data (X_train_label):")
print(" - dtype:", X_train_label.dtype)
print(" - shape:", X_train_label.shape)
print(" - memory used:", round(X_train_label.nbytes / 1024**2, 2), "MB")


# In[11]:


# Define feature_columns (if not already defined)
feature_columns = [
    'sex', 'age', 'height', 'weight', 'reported_cough_dur',
    'tb_prior', 'tb_prior_Pul', 'tb_prior_Extrapul', 'tb_prior_Unknown',
    'hemoptysis', 'heart_rate', 'temperature', 'weight_loss',
    'smoke_lweek', 'fever', 'night_sweats'
]


# In[ ]:


import pandas as pd

# Load your clinical data
df = pd.read_csv('CODA_TB_Clinical_Meta_Info.csv')

# Calculate gender counts and percentages
sex_counts = df['sex'].value_counts()
sex_percent = df['sex'].value_counts(normalize=True) * 100

# Calculate TB status counts
tb_counts = df['tb_status'].value_counts()
tb_percent = df['tb_status'].value_counts(normalize=True) * 100

# Function to get median and IQR
def median_iqr(series):
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"

# Create the summary table
summary = {
    "Sex": [
        f"Female: {sex_counts.get('Female', 0)} ({sex_percent.get('Female', 0):.1f}%)",
        f"Male: {sex_counts.get('Male', 0)} ({sex_percent.get('Male', 0):.1f}%)"
    ],
    "Age": median_iqr(df["age"]),
    "Height (cm)": median_iqr(df["height"]),
    "Weight (kg)": median_iqr(df["weight"]),
    "Duration of cough (days)": median_iqr(df["reported_cough_dur"]),
    "TB Status": [
        f"Negative (0): {tb_counts.get(0, 0)} ({tb_percent.get(0, 0):.1f}%)",
        f"Positive (1): {tb_counts.get(1, 0)} ({tb_percent.get(1, 0):.1f}%)"
    ]
}

# Display it nicely
print("Participant Demographics Summary:")
for key, value in summary.items():
    if isinstance(value, list):
        print(f"{key}:")
        for v in value:
            print(f"  - {v}")
    else:
        print(f"{key}: {value}")


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get index of 'weight' column
weight_index = feature_columns.index('weight')

# Extract values
raw_weights = train_df['weight'].values
scaled_weights = X_train_label[:, weight_index]

# Plot both
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(raw_weights, bins=30, kde=True)
plt.title("Original Weight (Before Scaling)")
plt.xlabel("Weight (kg)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(scaled_weights, bins=30, kde=True)
plt.title("Standardized Weight (After Scaling)")
plt.xlabel("Z-score")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Print sample comparisons
print("\nSample raw vs. scaled weight values:")
for raw, scaled in zip(raw_weights[:10], scaled_weights[:10]):
    print(f"Original: {raw:.1f} kg → Scaled: {scaled:.3f}")


# In[23]:


print("X_train_label shape:", X_train_label.shape)


# In[24]:


print("Number of feature columns:", len(feature_columns))


# In[25]:


weight_index = feature_columns.index('weight')

# Scaled value
scaled_weight = X_train_label[:, weight_index]

# Raw value from train_df
raw_weight = train_df['weight'].values

# Print sample
for raw, scaled in zip(raw_weight[:5], scaled_weight[:5]):
    print(f"Original: {raw:.1f} → Scaled: {scaled:.3f}")


# In[52]:


import pandas as pd

# Load clinical data
df = pd.read_csv("CODA_TB_Clinical_Meta_Info.csv")

# Total participants
total_n = df.shape[0]

# --- 1. Sex breakdown with TB status ---
# Count of sex by tb_status
sex_tb_counts = df.groupby(['sex', 'tb_status']).size().unstack(fill_value=0)

# Overall sex distribution
sex_total = df['sex'].value_counts()
sex_percent = (sex_total / total_n * 100).round(1)

# Format: "count (percentage%)"
sex_female = f"{sex_total['Female']} ({sex_percent['Female']}%)"
sex_male = f"{sex_total['Male']} ({sex_percent['Male']}%)"

# TB-positive/negative counts for each sex
female_tb0 = sex_tb_counts.loc['Female'][0]
female_tb1 = sex_tb_counts.loc['Female'][1]
male_tb0 = sex_tb_counts.loc['Male'][0]
male_tb1 = sex_tb_counts.loc['Male'][1]

# --- 2. Summary function ---
def median_iqr(series):
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"


    
# --- 3. Continuous feature summaries ---
age_summary = median_iqr(df['age'])
height_summary = median_iqr(df['height'])
weight_summary = median_iqr(df['weight'])
cough_summary = median_iqr(df['reported_cough_dur'])

# --- 4. Build summary table ---
table = pd.DataFrame([
    ["Sex", "Female", sex_female, f"TB-: {female_tb0}, TB+: {female_tb1}"],
    ["", "Male", sex_male, f"TB-: {male_tb0}, TB+: {male_tb1}"],
    ["Age", "Median", age_summary, ""],
    ["Height (cm)", "Median", height_summary, ""],
    ["Weight (kg)", "Median", weight_summary, ""],
    ["Duration of cough (days)", "Median", cough_summary, ""]
], columns=["Variable", "Category", f"DataSet (N={total_n})", "TB Status Breakdown"])

# Display
print(table.to_string(index=False))


# In[53]:


# ✅ Sample rate and original duration summary (based on training ZIP files)
import zipfile
import soundfile as sf

zip_paths = ['solicated_coughs.zip', 'longitudinal_1.zip', 'longitudinal_2.zip']
sample_rates = set()
durations = []

for zip_path in zip_paths:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for name in zipf.namelist():
                if name.lower().endswith('.wav'):
                    try:
                        with zipf.open(name) as f:
                            audio, rate = sf.read(f)
                            sample_rates.add(rate)
                            durations.append(len(audio) / rate)
                    except Exception as e:
                        print(f"Error reading {name} in {zip_path}: {e}")
    except FileNotFoundError:
        print(f"❌ File not found: {zip_path}")

# ✅ Print results
print("\n🎯 Sample rates found in training audio:")
for sr in sample_rates:
    print(f" - {sr} Hz")

if durations:
    print("\n🎯 Duration of original audio files before fix_length():")
    print(f" - Min: {min(durations):.3f} sec")
    print(f" - Max: {max(durations):.3f} sec")
    print(f" - Avg: {sum(durations) / len(durations):.3f} sec")
else:
    print("⚠️ No durations could be read.")


# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and merge metadata files like in your training pipeline
clinical_data = pd.read_csv('CODA_TB_Clinical_Meta_Info.csv')
solicited_data = pd.read_csv('CODA_TB_Solicited_Meta_Info.csv')
longitudinal_data = pd.read_csv('CODA_TB_Longitudnal_Meta_Info.csv')

merged_solicited = pd.merge(solicited_data, clinical_data, on="participant")
merged_longitudinal = pd.merge(longitudinal_data, clinical_data, on="participant")
combined_data = pd.concat([merged_solicited, merged_longitudinal], ignore_index=True)

# Clinical features used in your training
feature_columns = [
    'sex', 'age', 'height', 'weight', 'reported_cough_dur', 'tb_prior', 'tb_prior_Pul',
    'tb_prior_Extrapul', 'tb_prior_Unknown', 'hemoptysis', 'heart_rate', 'temperature',
    'weight_loss', 'smoke_lweek', 'fever', 'night_sweats'
]

# Fill missing values (like you did in training)
for col in combined_data[feature_columns].select_dtypes(include=['object', 'category']).columns:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    combined_data[col] = encoder.fit_transform(combined_data[col].astype(str))

for col in feature_columns:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mean())

# Now fit the scaler
scaler = StandardScaler()
scaler.fit(combined_data[feature_columns])

print("Training Mean:", scaler.mean_)
print("Training Std:", scaler.scale_)


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


print("X_train dtype:", X_train.dtype)
print("Memory used by X_train:", X_train.nbytes / 1024**3, "GB")


#  ##  **Model Evaluation Functions**

# In[9]:


def evaluate_model_performance(model, X_test, X_test_label, y_test):
    """
    Evaluate model performance with various metrics
    """
    # Get predictions
    y_pred = model.predict([X_test, X_test_label])
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'auc': roc_auc_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes),
        'recall': recall_score(y_test, y_pred_classes)
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return metrics


# In[10]:


from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix, 
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model_performance(model, X_test, X_test_label, y_test):
    """
    Evaluate model performance with various metrics and visualizations
    """

    # Get predictions
    y_pred = model.predict([X_test, X_test_label])
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'auc': roc_auc_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes),
        'recall': recall_score(y_test, y_pred_classes)
    }

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.4f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure()
    plt.plot(recall, precision, label='PR Curve', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prediction Probabilities Histogram
    plt.figure()
    plt.hist(y_pred[y_test == 0], bins=30, alpha=0.6, label='Class 0', color='blue')
    plt.hist(y_pred[y_test == 1], bins=30, alpha=0.6, label='Class 1', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probabilities by Class')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve', color='purple')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Threshold vs. Metric Curve
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_test, (y_pred >= t).astype(int)) for t in thresholds]
    precisions = [precision_score(y_test, (y_pred >= t).astype(int)) for t in thresholds]
    recalls = [recall_score(y_test, (y_pred >= t).astype(int)) for t in thresholds]

    plt.figure()
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Decision Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return metrics


# ## **Feature Importance Analysis**

# In[6]:


def analyze_feature_importance(model, X_test, X_test_label, feature_columns):
    """
    Analyze and visualize feature importance using SHAP values
    """
    print("\nAnalyzing feature importance...")
    
    try:
        # Create background dataset for SHAP
        background_size = min(100, len(X_test))
        
        # Create explainer with a wrapper function
        def model_predict(clinical_features):
            batch_size = len(clinical_features)
            repeated_audio = np.repeat(X_test[0:1], batch_size, axis=0)
            predictions = model.predict([repeated_audio, clinical_features], batch_size=batch_size)
            return predictions.reshape(-1)
        
        # Use background samples
        background_data = X_test_label[:20]
        explainer = shap.KernelExplainer(
            model_predict, 
            background_data,
            link="logit"
        )
        
        # Calculate SHAP values
        test_samples = X_test_label[:background_size]
        shap_values = explainer.shap_values(test_samples, nsamples=100)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # 1. Summary Plot
        plt.figure(figsize=(10, 6), num='SHAP Summary Plot')
        shap.summary_plot(
            shap_values,
            test_samples,
            feature_names=feature_columns,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.show()

        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(10, 6), num='Feature Importance Ranking')
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importance Ranking')
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

        # 3. SHAP Values Distribution
        plt.figure(figsize=(12, 8), num='SHAP Values Distribution')
        shap.summary_plot(
            shap_values,
            test_samples,
            feature_names=feature_columns,
            plot_type="violin",
            show=False
        )
        plt.title('SHAP Values Distribution')
        plt.tight_layout()
        plt.show()

        # Print feature importance summary
        #print("\nFeature Importance Summary:")
        #for feature, importance in zip(feature_columns, importance_df['Importance']):
           #print(f"{feature}: {importance:.4f}")
        
      #  CHANGED HERE
        # Print feature importance summary using the sorted order
        print("\nFeature Importance Summary (sorted by mean |SHAP|):")
        for _, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        return importance_df
        
    except Exception as e:
        print(f"Error in feature importance analysis: {str(e)}")
        traceback.print_exc()
        return None

def plot_individual_prediction_explanation(model, X_sample, X_sample_label, feature_columns):
    """
    Explain individual prediction using SHAP
    """
    try:
        # Get prediction with proper reshaping
        pred = model.predict([
            X_sample.reshape(1, -1, 1), 
            X_sample_label.reshape(1, -1)
        ])[0][0]
        
        # Create explainer with proper prediction function
        def model_predict(clinical_features):
            batch_size = len(clinical_features)
            repeated_audio = np.repeat(X_sample.reshape(1, -1, 1), batch_size, axis=0)
            predictions = model.predict([repeated_audio, clinical_features], batch_size=batch_size)
            return predictions.reshape(-1)
        
        # Create explainer with more background samples
        background_data = np.repeat(X_sample_label.reshape(1, -1), 20, axis=0)
        explainer = shap.KernelExplainer(
            model_predict, 
            background_data,
            link="logit"
        )
        
        # Calculate SHAP values with more samples
        shap_values = explainer.shap_values(
            X_sample_label.reshape(1, -1), 
            nsamples=100
        )
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()
        
        # Print raw values for debugging
        print("\nRaw SHAP values:")
        for feature, value in zip(feature_columns, shap_values):
            print(f"{feature}: {value:.6f}")
        
    
        
        return pd.DataFrame({
            'Feature': feature_columns,
            'Impact': shap_values,
            'Abs_Impact': np.abs(shap_values)
        }).sort_values('Abs_Impact', ascending=False)
        
    except Exception as e:
        print(f"Error in individual prediction explanation: {str(e)}")
        traceback.print_exc()
        return None


# ## **plot_benchmark_results**

# In[11]:


def plot_benchmark_results(benchmark_results, benchmark_histories, cnnbasedNet_metrics, cnnbasedNet_history):
    """Plot comparison of all models including CNNbasedNet"""
    # Add CNNbasedNet results to benchmark results
    benchmark_results['CNNbasedNet'] = cnnbasedNet_metrics
    benchmark_histories['CNNbasedNet'] = cnnbasedNet_history
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    accuracies = {name: metrics['accuracy'] for name, metrics in benchmark_results.items()}
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=axes[0, 0])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    
    # AUC comparison
    aucs = {name: metrics['auc'] for name, metrics in benchmark_results.items()}
    sns.barplot(x=list(aucs.keys()), y=list(aucs.values()), ax=axes[0, 1])
    axes[0, 1].set_title('Model AUC Comparison')
    axes[0, 1].set_ylabel('AUC')
    
    # Training history comparison
    for name, history in benchmark_histories.items():
        axes[1, 0].plot(history['accuracy'], label=f'{name}')
    axes[1, 0].set_title('Training Accuracy History')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    
    # Inference time comparison
    inference_times = {name: metrics['inference_time'] for name, metrics in benchmark_results.items()}
    sns.barplot(x=list(inference_times.keys()), y=list(inference_times.values()), ax=axes[1, 1])
    axes[1, 1].set_title('Inference Time Comparison')
    axes[1, 1].set_ylabel('Time (s)')
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(benchmark_results).T



def plot_model_comparisons(models_results, histories, X_test, y_test):
    """
    Create comprehensive comparison visualizations for all models
    """
    # 1. ROC Curves
    plt.figure(figsize=(10, 6), num='ROC Curves Comparison')
    for name, metrics in models_results.items():
        if name == 'CNNbasedNet121':
            y_pred = model.predict([X_test, X_test_label])
        else:
            y_pred = models_results[name]['predictions']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, len(models_results), figsize=(15, 4), num='Confusion Matrices')
    for i, (name, metrics) in enumerate(models_results.items()):
        if name == 'CNNbasedNet121':
            y_pred = (model.predict([X_test, X_test_label]) > 0.5).astype(int)
        else:
            y_pred = (models_results[name]['predictions'] > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 6), num='Precision-Recall Curves')
    for name, metrics in models_results.items():
        if name == 'CNNbasedNet121':
            y_pred = model.predict([X_test, X_test_label])
        else:
            y_pred = models_results[name]['predictions']
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 4. Training History Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), num='Training History')
    
    # Accuracy
    for name, history in histories.items():
        ax1.plot(history['accuracy'], label=f'{name} (Train)')
        ax1.plot(history['val_accuracy'], label=f'{name} (Val)', linestyle='--')
    ax1.set_title('Accuracy History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    for name, history in histories.items():
        ax2.plot(history['loss'], label=f'{name} (Train)')
        ax2.plot(history['val_loss'], label=f'{name} (Val)', linestyle='--')
    ax2.set_title('Loss History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # AUC
    for name, history in histories.items():
        if 'auc' in history:
            ax3.plot(history['auc'], label=f'{name} (Train)')
            ax3.plot(history['val_auc'], label=f'{name} (Val)', linestyle='--')
    ax3.set_title('AUC History')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.legend()
    ax3.grid(True)
    
    # Learning Rate
    for name, history in histories.items():
        if 'lr' in history:
            ax4.plot(history['lr'], label=name)
    ax4.set_title('Learning Rate History')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nModel Performance Summary:")
    print("-" * 50)
    for name, metrics in models_results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {metrics['auc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Inference Time: {metrics['inference_time']*1000:.2f} ms/sample")



# In[12]:


def plot_benchmark_results(benchmark_results, benchmark_histories, cnnbasedNet_metrics, cnnbasedNet_history):
    """Plot comparison of all models including CNNbasedNet"""
    # Add CNNbasedNet results to benchmark results
    benchmark_results['CNNbasedNet'] = cnnbasedNet_metrics
    benchmark_histories['CNNbasedNet'] = cnnbasedNet_history
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    accuracies = {name: metrics['accuracy'] for name, metrics in benchmark_results.items()}
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=axes[0, 0])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    
    # AUC comparison
    aucs = {name: metrics['auc'] for name, metrics in benchmark_results.items()}
    sns.barplot(x=list(aucs.keys()), y=list(aucs.values()), ax=axes[0, 1])
    axes[0, 1].set_title('Model AUC Comparison')
    axes[0, 1].set_ylabel('AUC')
    
    # Training history comparison
    for name, history in benchmark_histories.items():
        axes[1, 0].plot(history['accuracy'], label=f'{name}')
    axes[1, 0].set_title('Training Accuracy History')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    
    # Inference time comparison
    inference_times = {name: metrics['inference_time'] for name, metrics in benchmark_results.items()}
    sns.barplot(x=list(inference_times.keys()), y=list(inference_times.values()), ax=axes[1, 1])
    axes[1, 1].set_title('Inference Time Comparison')
    axes[1, 1].set_ylabel('Time (s)')
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(benchmark_results).T



def plot_model_comparisons(models_results, histories, X_test, y_test):
    """
    Create comprehensive comparison visualizations for all models
    """
    # 1. ROC Curves
    plt.figure(figsize=(10, 6), num='ROC Curves Comparison')
    for name, metrics in models_results.items():
        if name == 'CNNbasedNet121':
            y_pred = model.predict([X_test, X_test_label])
        else:
            y_pred = models_results[name]['predictions']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, len(models_results), figsize=(15, 4), num='Confusion Matrices')
    for i, (name, metrics) in enumerate(models_results.items()):
        if name == 'CNNbasedNet121':
            y_pred = (model.predict([X_test, X_test_label]) > 0.5).astype(int)
        else:
            y_pred = (models_results[name]['predictions'] > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 6), num='Precision-Recall Curves')
    for name, metrics in models_results.items():
        if name == 'CNNbasedNet121':
            y_pred = model.predict([X_test, X_test_label])
        else:
            y_pred = models_results[name]['predictions']
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 4. Training History Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), num='Training History')
    
    # Accuracy
    for name, history in histories.items():
        ax1.plot(history['accuracy'], label=f'{name} (Train)')
        ax1.plot(history['val_accuracy'], label=f'{name} (Val)', linestyle='--')
    ax1.set_title('Accuracy History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    for name, history in histories.items():
        ax2.plot(history['loss'], label=f'{name} (Train)')
        ax2.plot(history['val_loss'], label=f'{name} (Val)', linestyle='--')
    ax2.set_title('Loss History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # AUC
    for name, history in histories.items():
        if 'auc' in history:
            ax3.plot(history['auc'], label=f'{name} (Train)')
            ax3.plot(history['val_auc'], label=f'{name} (Val)', linestyle='--')
    ax3.set_title('AUC History')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.legend()
    ax3.grid(True)
    
    # Learning Rate
    for name, history in histories.items():
        if 'lr' in history:
            ax4.plot(history['lr'], label=name)
    ax4.set_title('Learning Rate History')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nModel Performance Summary:")
    print("-" * 50)
    for name, metrics in models_results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {metrics['auc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Inference Time: {metrics['inference_time']*1000:.2f} ms/sample")



# In[13]:


def optimize_model(X_train, y_train, X_val, y_val):
    """Perform hyperparameter optimization"""
    tuner = kt.Hyperband(
        create_hyperparameter_tuning_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='cnnbasedNet_optimization'
    )
    
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=[stop_early]
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nOptimal Hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    
    return best_hps

def preprocess_data(X_train_label, X_val_label, feature_columns):
    """
    Preprocess the clinical features data
    """
    # Create scaler
    scaler = StandardScaler()
    
    # Fit on training data and transform both training and test
    X_train_scaled = scaler.fit_transform(X_train_label)
    X_val_scaled = scaler.transform(X_val_label)
    
    # Convert to DataFrame for easier handling
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_val_df = pd.DataFrame(X_val_scaled, columns=feature_columns)
    
    return X_train_df, X_val_df, scaler

def plot_feature_distributions(data, feature_name, ax, title):
    """
    Helper function to plot feature distributions
    """
    sns.histplot(data=data, ax=ax, kde=True)
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Count')

def create_feature_plots(X_train_label, X_val_label, feature_columns):
    """
    Create comprehensive feature analysis plots
    """
    n_features = len(feature_columns)
    fig, axes = plt.subplots(3, n_features, figsize=(20, 12))
    
    for i, feature in enumerate(feature_columns):
        # Training distribution
        plot_feature_distributions(
            X_train_label[:, i],
            feature,
            axes[0, i],
            f'{feature}\nTraining Distribution'
        )
        
        # Test distribution
        plot_feature_distributions(
            X_val_label[:, i],
            feature,
            axes[1, i],
            f'{feature}\nValidation Distribution'
        )
        
        # Box plot comparison
        sns.boxplot(
            data=[X_train_label[:, i], X_val_label[:, i]],
            ax=axes[2, i]
        )
        axes[2, i].set_xticklabels(['Train', 'Validation'])
        axes[2, i].set_title(f'{feature}\nDistribution Comparison')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(data, feature_columns):
    """
    Create correlation matrix heatmap
    """
    corr_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        xticklabels=feature_columns,
        yticklabels=feature_columns
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_preprocessing_comparison(X_raw, X_processed, feature_columns):
    """
    Compare distributions before and after preprocessing
    """
    n_features = len(feature_columns)
    fig, axes = plt.subplots(2, n_features, figsize=(20, 8))
    
    for i, feature in enumerate(feature_columns):
        # Before preprocessing
        sns.histplot(
            data=X_raw[:, i],
            ax=axes[0, i],
            kde=True,
            color='red',
            alpha=0.5
        )
        axes[0, i].set_title(f'{feature} (Before)')
        
        # After preprocessing
        sns.histplot(
            data=X_processed[:, i],
            ax=axes[1, i],
            kde=True,
            color='blue',
            alpha=0.5
        )
        axes[1, i].set_title(f'{feature} (After)')
    
    plt.tight_layout()
    return fig


# In[14]:


def create_hyperparameter_tuning_model(hp):
    """
    Create model with tunable hyperparameters
    """
    try:
        length = 22050
        inputs = Input(shape=(length, 1))
        
        # Tunable initial filters
        filters = hp.Int('initial_filters', min_value=16, max_value=64, step=16)
        
        # First CNN block with tunable parameters
        x = Conv1D(
            filters=filters,
            kernel_size=hp.Int('kernel_1', 32, 128, step=32),
            strides=hp.Int('strides_1', 2, 4, step=1),
            padding='same'
        )(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=hp.Float('leaky_alpha', 0.1, 0.3, step=0.1))(x)
        x = MaxPooling1D(pool_size=hp.Int('pool_1', 4, 8, step=2))(x)
        x = Dropout(hp.Float('dropout_1', 0.1, 0.3, step=0.1))(x)
        
        # Second CNN block
        x = Conv1D(
            filters=filters * 2,
            kernel_size=hp.Int('kernel_2', 16, 64, step=16),
            strides=2,
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=hp.Float('leaky_alpha_2', 0.1, 0.3, step=0.1))(x)
        x = MaxPooling1D(pool_size=hp.Int('pool_2', 4, 8, step=2))(x)
        x = Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.1))(x)
        
        # Dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(
            units=hp.Int('dense_units', 128, 512, step=128),
            activation='relu'
        )(x)
        x = Dropout(hp.Float('dropout_dense', 0.3, 0.6, step=0.1))(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Tunable learning rate
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC()]
        )
        
        return model
    except Exception as e:
        print(f"Error in create_hyperparameter_tuning_model: {str(e)}")
        traceback.print_exc()
        return None

def apply_data_augmentation(X, y, augmentation_factor=2):
    """
    Apply data augmentation techniques for audio data
    """
    try:
        X_aug = []
        y_aug = []
        
        for i in tqdm(range(len(X)), desc="Applying augmentation"):
            # Original sample
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Add noise
            if np.random.random() < 0.5:
                noise_factor = np.random.uniform(0.001, 0.02)
                noise = np.random.normal(0, noise_factor, X[i].shape)
                X_aug.append(X[i] + noise)
                y_aug.append(y[i])
            
            # Time stretching
            if np.random.random() < 0.5:
                stretch_factor = np.random.uniform(0.8, 1.2)
                stretched = librosa.effects.time_stretch(X[i].flatten(), stretch_factor)
                X_aug.append(stretched[:len(X[i])].reshape(X[i].shape))
                y_aug.append(y[i])
            
            # Pitch shifting
            if np.random.random() < 0.5:
                n_steps = np.random.randint(-2, 3)
                shifted = librosa.effects.pitch_shift(X[i].flatten(), sr=22050, n_steps=n_steps)
                X_aug.append(shifted.reshape(X[i].shape))
                y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)
    except Exception as e:
        print(f"Error in apply_data_augmentation: {str(e)}")
        traceback.print_exc()
        return X, y

def optimize_inference(model):
    """
    Optimize model for inference
    """
    try:
        # Convert to TF Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        # Save TF Lite model
        os.makedirs('./models', exist_ok=True)
        with open('./models/model_optimized.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Convert to ONNX
        try:
            import tf2onnx
            spec = (tf.TensorSpec((None, 22050, 1), tf.float32, name="input"),)
            output_path = "./models/model_optimized.onnx"
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
            print("ONNX model saved successfully")
        except Exception as e:
            print(f"Error converting to ONNX: {str(e)}")
        
        return tflite_model
    except Exception as e:
        print(f"Error in optimize_inference: {str(e)}")
        traceback.print_exc()
        return None


# 
# ## **Main Execution**
# 

# ## **1. Load and preprocess data**

# In[20]:


print("Starting data loading and preprocessing...")
X_train, X_val, X_test, X_train_label, X_val_label, X_test_label, y_train, y_val, y_test = load_and_preprocess_data()


# In[51]:


if __name__ == "__main__":
    print("🔄 Starting full preprocessing pipeline...")
    
    X_train, X_val, X_test, X_train_label, X_val_label, X_test_label, y_train, y_val, y_test = load_and_preprocess_data()

    print("✅ Data successfully loaded.")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")


# In[19]:


import os
print(os.getcwd())


# In[ ]:





# In[ ]:





# ## **2.Visualizing data distributions and relationships**

# In[13]:


selected_features = ['weight', 'heart_rate', 'smoke_lweek', 'night_sweats']

# 📊 Subsection 1: Before Preprocessing
visualize_raw_distributions_separate(train_raw, val_raw, selected_features, feature_columns)

# 🔧 Subsection 2: After Preprocessing
visualize_scaled_distributions_separate(X_train_label, X_val_label, selected_features, feature_columns)


# In[ ]:





# In[ ]:





# ## **3.Train model**

# In[22]:


# 3. Train model
print("\nStarting model training...")
model, history = train_cnnbasedNet(
    X_train, X_train_label, y_train,
    X_val, X_val_label, y_val  # ✅ changed from test to validation
)


# ### **4.Plot training history**

# In[23]:


# 4. Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# ### **5 evaluate model performance**

# In[24]:


# 5. Evaluate model performance
print("\nEvaluating model performance...")
metrics = evaluate_model_performance(model, X_test, X_test_label, y_test)


# In[ ]:





# In[ ]:





# In[25]:


print("\nEvaluating model performance...")
metrics = evaluate_model_performance(model, X_test, X_test_label, y_test)


# In[26]:


# 5. Evaluate model performance
print("\nEvaluating model performance...")
metrics = evaluate_model_performance(model, X_test, X_test_label, y_test)


# ### **6.Save results**

# In[27]:


# 6. Save results
print("\nSaving results...")
model.save('./models/best_model.h5')

with open('./results/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
    
with open('./results/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)


# In[ ]:





# In[30]:


import tensorflow as tf
import os

def save_model_all_formats(model, input_shape=(22050, 1), model_name="best_model"):
    """
    Save a trained Keras model in multiple formats:
    - .keras (recommended)
    - .h5 (legacy)
    - SavedModel directory
    - .tflite (for mobile/embedded)
    - .onnx (for cross-framework support)
    """
    os.makedirs("./exported_models", exist_ok=True)

    print("🔄 Saving model in all formats...")

    # ✅ Save in Keras recommended format (.keras)
    keras_path = f"./exported_models/{model_name}.keras"
    model.save(keras_path)
    print(f"✅ Saved: {keras_path}")

    # ✅ Save in HDF5 format (.h5)
    h5_path = f"./exported_models/{model_name}.h5"
    model.save(h5_path)
    print(f"✅ Saved: {h5_path}")

    # ✅ Save in SavedModel format (directory)
    saved_model_path = f"./exported_models/{model_name}_savedmodel"
    model.save(saved_model_path)
    print(f"✅ Saved: {saved_model_path}/")

    # ✅ Save as TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = f"./exported_models/{model_name}.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ Saved: {tflite_path}")
    except Exception as e:
        print(f"⚠️ Failed to save TFLite model: {e}")

    # ✅ Save as ONNX
    try:
        import tf2onnx
        spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
        onnx_path = f"./exported_models/{model_name}.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
        print(f"✅ Saved: {onnx_path}")
    except Exception as e:
        print(f"⚠️ Failed to save ONNX model: {e}")


# In[32]:


import tf2onnx
import tensorflow as tf

spec = (
    tf.TensorSpec((None, 22050, 1), tf.float32, name="audio_input"),
    tf.TensorSpec((None, 16), tf.float32, name="clinical_input")
)

onnx_path = "./exported_models/tb_detector_final.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=onnx_path
)

print(f"✅ Saved: {onnx_path}")


# In[33]:


import zipfile
import os

def zip_exported_models(zip_name="tb_detector_models.zip", folder="./exported_models"):
    zip_path = os.path.join(folder, "..", zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=folder)
                zipf.write(full_path, arcname=rel_path)
    print(f"✅ All model files zipped to: {zip_path}")


# In[34]:


zip_exported_models()


# In[20]:


from tensorflow.keras.models import load_model

model = load_model("best_model.h5")


# In[39]:


from tensorflow.keras.models import load_model
import pickle

from tensorflow.keras.models import load_model
import pickle
import time

# Load model
model = load_model('./models/best_model.h5')

# Load training history
with open('./results/training_history.pkl', 'rb') as f:
    history_data = pickle.load(f)  # this replaces history.history

# (Optional) Load saved final metrics too
with open('./results/metrics.pkl', 'rb') as f:
    saved_metrics = pickle.load(f)


# ### **print Analyzed feature importance**

# In[40]:


# Add feature importance analysis
print("\nAnalyzing feature importance...")
feature_columns = [
            'sex',
            'age',
            'height',
            'weight',
            'reported_cough_dur',
            'tb_prior',
            'tb_prior_Pul',
            'tb_prior_Extrapul',
            'tb_prior_Unknown',
            'hemoptysis',
            'heart_rate',
            'temperature',
            'weight_loss',
            'smoke_lweek',
            'fever',
            'night_sweats'
]

importance_results = analyze_feature_importance(
    model, X_test, X_test_label, feature_columns
)

print("\nTraining and evaluation complete!")


# ### **Model Benchmarking**

# In[23]:


##############################################################################
# Model Benchmarking
##############################################################################

def create_resnet50(length=22050, seed=42):
    """LightResNet: Minimal residual block with audio + clinical input"""
    audio_input = Input(shape=(length, 1))
    clinical_input = Input(shape=(16,))

    # Audio Conv block
    x = Conv1D(16, kernel_size=9, strides=2, padding='same', kernel_initializer=GlorotUniform(seed))(audio_input)
    x = BatchNormalization()(x)
    shortcut = x  # identity skip
    x = Conv1D(16, kernel_size=3, padding='same', kernel_initializer=GlorotUniform(seed+1))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = GlobalAveragePooling1D()(x)

    # Clinical input
    y = Dense(8, kernel_initializer=GlorotUniform(seed+2))(clinical_input)

    # Merge
    z = Concatenate()([x, y])
    z = Dense(16)(z)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[audio_input, clinical_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', AUC()])
    return model



def create_efficientnet(length=22050, seed=42):
    """LightEfficient: Minimal squeeze-excite block with audio + clinical input"""

    def squeeze_excite_block(x, ratio=4):
        filters = x.shape[-1]
        se = GlobalAveragePooling1D()(x)
        se = Dense(filters // ratio)(se)
        se = Dense(filters, activation='sigmoid')(se)
        return Multiply()([x, se])

    audio_input = Input(shape=(length, 1))
    clinical_input = Input(shape=(16,))

    # Audio Conv block
    x = Conv1D(16, kernel_size=9, strides=2, padding='same', kernel_initializer=GlorotUniform(seed))(audio_input)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x)
    x = GlobalAveragePooling1D()(x)

    # Clinical input
    y = Dense(8, kernel_initializer=GlorotUniform(seed+2))(clinical_input)

    # Merge
    z = Concatenate()([x, y])
    z = Dense(16)(z)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[audio_input, clinical_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', AUC()])
    return model

def benchmark_models(X_train, X_train_label, y_train,
                     X_val, X_val_label, y_val,
                     epochs=50):
    """
    Train and evaluate LightResNet and LightEfficientNet models
    using training and validation data (not test).
    """
    results = {}
    histories = {}
    inference_times = {}

    models = {
        'LightResNet': create_resnet50(length=22050, seed=42),
        'LightEfficient': create_efficientnet(length=22050, seed=42)
    }

    batch_size = 16
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = model.fit(
            [X_train, X_train_label],
            y_train,
            validation_data=([X_val, X_val_label], y_val),  # ✅ validation set
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ],
            verbose=1
        )

        # Measure inference time on validation set
        start_time = time.time()
        y_pred = model.predict([X_val, X_val_label], batch_size=batch_size)
        inference_time = (time.time() - start_time) / len(X_val)

        y_pred_classes = (y_pred > 0.5).astype(int)
        results[name] = {
            'accuracy': accuracy_score(y_val, y_pred_classes),
            'precision': precision_score(y_val, y_pred_classes),
            'recall': recall_score(y_val, y_pred_classes),
            'f1': f1_score(y_val, y_pred_classes),
            'auc': roc_auc_score(y_val, y_pred),
            'inference_time': inference_time,
            'predictions': y_pred
        }

        histories[name] = history.history
        inference_times[name] = inference_time
        tf.keras.backend.clear_session()

    return results, histories, inference_times


# In[26]:


benchmark_results, benchmark_histories, inference_times = benchmark_models(
    X_train=X_train,
    X_train_label=X_train_label,
    y_train=y_train,
    X_val=X_val,
    X_val_label=X_val_label,
    y_val=y_val,
    epochs=50
)


# In[ ]:


# Set history to None since you didn’t save it
cnn_history = None


# In[25]:


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Add, Dense, Dropout, GlobalAveragePooling1D,
    Multiply, Concatenate
)
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ##  **Plot_benchmark_results**

# In[27]:


# Fix the model name if saved incorrectly
if "MetforNet" in benchmark_results:
    benchmark_results["CNNbasedNet"] = benchmark_results.pop("MetforNet")
    benchmark_histories["CNNbasedNet"] = benchmark_histories.pop("MetforNet")


# In[ ]:





# In[28]:


from tensorflow.keras.models import load_model
import pickle

from tensorflow.keras.models import load_model
import pickle
import time

# Load model
model = load_model('./models/best_model.h5')

# Load training history
with open('./results/training_history.pkl', 'rb') as f:
    history_data = pickle.load(f)  # this replaces history.history

# (Optional) Load saved final metrics too
with open('./results/metrics.pkl', 'rb') as f:
    saved_metrics = pickle.load(f)


# In[ ]:





# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import os


def plot_benchmark_results(results, histories, cnnbasedNet_metrics, cnnbasedNet_history=None, y_test=None, save_dir="./benchmark_plots"):
    """
    Visualize benchmark results: bar plots of metrics, validation loss, ROC curves, and training accuracy.

    Parameters:
        results: dict of metrics per model (from benchmark_models)
        histories: dict of training history per model
        cnnbasedNet_metrics: dict of pre-evaluated CNNbasedNet metrics
        cnnbasedNet_history: optional dict of training history for CNNbasedNet
        y_test: true labels (required for ROC curves)
        save_dir: directory to save all plots and summary CSV/LaTeX

    Returns:
        benchmark_df: pandas DataFrame of all model metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    # Combine all metrics into a DataFrame
    benchmark_df = pd.DataFrame(results).T
    benchmark_df.loc['CNNbasedNet'] = cnnbasedNet_metrics

    # Convert only numeric columns to float, skip 'predictions'
    numeric_df = benchmark_df.drop(columns=['predictions']).astype(float)
    benchmark_df.update(numeric_df)

    # Compute ROC AUC scores and store them
    if y_test is not None:
        for model_name in benchmark_df.index:
            if 'predictions' in results[model_name]:
                y_score = results[model_name]['predictions']
                fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_score)
                model_auc = auc(fpr, tpr)
                benchmark_df.at[model_name, 'auc'] = model_auc

    # Print numerical results
    print("\nBenchmark Results (Numeric):")
    print(numeric_df)

    # Save numeric results to CSV
    csv_path = os.path.join(save_dir, "benchmark_metrics.csv")
    numeric_df.to_csv(csv_path)


    # Save numeric results to LaTeX table
    latex_path = os.path.join(save_dir, "benchmark_metrics.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(numeric_df.to_latex(float_format="{:.4f}".format))
        f.write("\\caption{Performance metrics comparison of CNNbasedNet and baseline models.}\n")
        f.write("\\label{tab:benchmark}\n")
        f.write("\\end{table}\n")
   
    # Bar plot of metrics
    plt.figure(figsize=(12, 6))
    numeric_df.drop(columns=['inference_time']).plot(kind='bar')
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_comparison.png"))
    plt.show()

    # Inference time comparison
    plt.figure(figsize=(8, 5))
    benchmark_df['inference_time'].plot(kind='bar', color='darkcyan')
    plt.title("Average Inference Time per Sample (seconds)")
    plt.ylabel("Seconds")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "inference_time.png"))
    plt.show()

    # Validation loss curves
    plt.figure()
    combined_histories = dict(histories)
    if cnnbasedNet_history is not None:
        combined_histories['CNNbasedNet'] = cnnbasedNet_history

    for model_name, history in combined_histories.items():
        if history and 'val_loss' in history:
            plt.plot(history['val_loss'], label=f'{model_name} Val Loss')

    plt.title("Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "validation_loss.png"))
    plt.show()

    # Training accuracy curves
    plt.figure()
    for model_name, history in combined_histories.items():
        if history and 'accuracy' in history:
            plt.plot(history['accuracy'], label=f'{model_name} Training Accuracy')

    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
    plt.show()

    # ROC Curves
    if y_test is not None:
        plt.figure(figsize=(8, 6))
        for model_name in benchmark_df.index:
            if 'predictions' in results[model_name]:
                y_score = results[model_name]['predictions']
                fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_score)
                model_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_comparison.png"))
        plt.show()

    return benchmark_df


# In[36]:


### the new  benchmark results 
# Plot benchmark results including CNNbasedNet
benchmark_df = plot_benchmark_results(
    benchmark_results, 
    benchmark_histories,
    CNNbasedNet_metrics,
    history_data  
)




# In[46]:


import tensorflow as tf

print(model.summary())  # Check model architecture


# In[47]:


print("Model weights before prediction:", model.get_weights()[0])  # Check if weights exist


# ## **end of the CODE**

# In[ ]:





# In[ ]:




