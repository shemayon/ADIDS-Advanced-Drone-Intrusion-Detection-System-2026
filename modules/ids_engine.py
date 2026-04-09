import tensorflow as tf
from tensorflow.keras import layers, models

def build_ids_model(input_dim):
    """
    Builds the Temporal-Spatial Lightweight Transformer (TSLT-Net) model.
    
    Args:
        input_dim (int): Number of input features.
        
    Returns:
        tf.keras.Model: The compiled or uncompiled Keras model.
    """
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    
    # Spatial projection
    x = layers.Dense(128, activation='relu', name="spatial_dense")(inputs)
    
    # Temporal reshaping (Spatial-to-Temporal transition)
    # 128 units reshaped into a 16-step sequence with 8 features each
    x = layers.Reshape((16, 8), name="temporal_reshape")(x)
    
    # Normalization
    x = layers.LayerNormalization(name="layer_norm")(x)
    
    # Multi-Head Attention (Core Transformer mechanism)
    # Analyzes dependencies across the temporal sequence
    x = layers.MultiHeadAttention(num_heads=2, key_dim=4, name="mha_transformer")(x, x)
    
    # Global pooling to collapse temporal dimension
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
    
    model = models.Model(inputs, outputs, name="TSLT_Net_Engine")
    
    return model

if __name__ == "__main__":
    # Test model build
    dummy_input_dim = 62
    model = build_ids_model(dummy_input_dim)
    model.summary()
    print("\nA-DIDS Core Engine: TSLT-Net built successfully.")
