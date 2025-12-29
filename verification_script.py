import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from classifiers.tcn import FusionModel
import matplotlib.pyplot as plt
import numpy as np

def visualize_verification(inputs, output, activations=None, filename='verification_plot.png'):
    """Visualizes input modalities and model output."""
    num_inputs = len(inputs)

    # Calculate total plots: Inputs + Encoders (if present) + Fused (if present) + Output
    num_plots = len(inputs) + 1 # Inputs + Output
    if activations:
        # One for each encoder output
        num_plots += len([k for k in activations if 'encoder' in k])
        # One for fused representation
        if 'fused' in activations:
            num_plots += 1
    
    plt.figure(figsize=(15, 4 * num_plots))
    
    # Plot Inputs
    # Plot Inputs and Activations
    current_plot = 1
    
    # Calculate global min/max for heatmaps to share color scale if desired, 
    # but individual scaling is usually better for visibility. 
    
    for i, (name, data) in enumerate(inputs.items()):
        plt.subplot(num_plots, 1, current_plot)
        current_plot += 1
        
        # Take the first sample in the batch
        sample_data = data[0].detach().cpu().numpy()
        
        if sample_data.ndim == 2: # (seq_len, dim)
            plt.imshow(sample_data.T, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
            plt.colorbar(label='Value')
            plt.ylabel(f'{name} Dim')
            plt.xlabel('Time Step')
            plt.title(f'Input: {name} (Shape: {sample_data.shape})')
        else:
             plt.plot(sample_data)
             plt.title(f'Input: {name} (Shape: {sample_data.shape})')

    # Plot Encoder Activations
    if activations:
        for name, data in activations.items():
            if 'encoder' in name:
                plt.subplot(num_plots, 1, current_plot)
                current_plot += 1
                
                # Take first sample: [seq_len, embedding_dim]
                sample_act = data[0].detach().cpu().numpy()
                
                plt.imshow(sample_act.T, aspect='auto', interpolation='nearest', origin='lower', cmap='magma')
                plt.colorbar(label='Activation')
                plt.ylabel('Emb Dim')
                plt.xlabel('Time Step')
                plt.title(f'Activation: {name} (Shape: {sample_act.shape})')
        
        # Plot Fused
        if 'fused' in activations:
            plt.subplot(num_plots, 1, current_plot)
            current_plot += 1
            
            sample_fused = activations['fused'][0].detach().cpu().numpy()
            
            plt.imshow(sample_fused.T, aspect='auto', interpolation='nearest', origin='lower', cmap='inferno')
            plt.colorbar(label='Activation')
            plt.ylabel('Fused Dim')
            plt.xlabel('Time Step')
            plt.title(f'Activation: Fused (Shape: {sample_fused.shape})')

    # Plot Output
    plt.subplot(num_plots, 1, num_plots)
    # Take first sample
    output_data = output[0].detach().cpu().numpy() # (seq_len, num_classes)
    
    for cls in range(output_data.shape[1]):
        plt.plot(output_data[:, cls], label=f'Class {cls}')
        
    plt.legend()
    plt.title(f'Model Output (Shape: {output_data.shape})')
    plt.xlabel('Time Step')
    plt.ylabel('Logits')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nVisualization saved to {filename}")
    plt.close()

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
        # Request activations
        output, activations = model(inputs, return_activations=True)
        print(f"Output Shape: {output.shape}")
        
        expected_shape = (batch_size, seq_len, num_classes)
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        
        print("\nSUCCESS: Dimension check passed!")
        
        # Visualize
        visualize_verification(inputs, output, activations=activations)
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
