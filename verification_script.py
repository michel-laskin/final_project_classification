import torch
from feature_dimension_eq.models import FusionModel

def test_model():
    print("Initializing Dummy Data...")
    batch_size = 4
    seq_len = 100
    
    # Define input dimensions
    input_dims = {
        'scalars': 10,
        'vectors_A': 50,
        'vectors_B': 12  # Adding another hypothetical modality
    }
    
    # Create random tensors
    inputs = {}
    for name, dim in input_dims.items():
        inputs[name] = torch.randn(batch_size, seq_len, dim)
        print(f"Created {name} with shape {inputs[name].shape}")

    # Model Params
    embedding_dim = 64
    tcn_channels = [64, 128, 64] # 3 layers
    num_classes = 2
    
    print("\nInstantiating FusionModel...")
    model = FusionModel(
        input_dims=input_dims,
        embedding_dim=embedding_dim,
        tcn_channels=tcn_channels,
        num_classes=num_classes
    )
    print(model)
    
    print("\nRunning Forward Pass...")
    try:
        output = model(inputs)
        print(f"Output Shape: {output.shape}")
        
        expected_shape = (batch_size, seq_len, num_classes)
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        
        print("\nSUCCESS: Dimension check passed!")
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
