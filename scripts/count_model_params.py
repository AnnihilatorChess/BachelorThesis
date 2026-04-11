"""
Calculates the parameter count for different models in The Well benchmark.
"""
import sys
import torch
from the_well.benchmark.models import FNO, UNetClassic, CNO

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/count_model_params.py <model_name>")
        print("Available models: fno, unet_classic, cno_default, cno_tuned, cno_tuned_large")
        sys.exit(1)
        
    model_name = sys.argv[1]
    
    # Dummy dimensions based on a typical 2D dataset
    DIM_IN = 4  # e.g., 4 input time steps of a single field
    DIM_OUT = 1 # e.g., 1 output time step of a single field
    N_SPATIAL_DIMS = 2
    SPATIAL_RESOLUTION = (128, 128)

    print(f"Calculating parameters for model: {model_name}")
    print(f"Input dims: dim_in={DIM_IN}, dim_out={DIM_OUT}, resolution={SPATIAL_RESOLUTION}")

    model = None
    if model_name == 'fno':
        model = FNO(
            dim_in=DIM_IN,
            dim_out=DIM_OUT,
            n_spatial_dims=N_SPATIAL_DIMS,
            spatial_resolution=SPATIAL_RESOLUTION,
            modes1=16,
            modes2=16,
            hidden_channels=128
        )
    elif model_name == 'unet_classic':
        model = UNetClassic(
            dim_in=DIM_IN,
            dim_out=DIM_OUT,
            n_spatial_dims=N_SPATIAL_DIMS,
            spatial_resolution=SPATIAL_RESOLUTION,
            init_features=48
        )
    elif model_name == 'cno_default':
        model = CNO(
            dim_in=DIM_IN,
            dim_out=DIM_OUT,
            n_spatial_dims=N_SPATIAL_DIMS,
            spatial_resolution=SPATIAL_RESOLUTION,
            N_layers=3,
            N_res=4,
            N_res_neck=4,
            channel_multiplier=64,
            use_bn=True
        )
    elif model_name == 'cno_tuned':
        # This is the proposed new configuration
        model = CNO(
            dim_in=DIM_IN,
            dim_out=DIM_OUT,
            n_spatial_dims=N_SPATIAL_DIMS,
            spatial_resolution=SPATIAL_RESOLUTION,
            N_layers=3,
            N_res=4,
            N_res_neck=4,
            channel_multiplier=24, # Tuned value
            use_bn=True
        )
    elif model_name == 'cno_tuned_large':
        # Larger tuned configuration
        model = CNO(
            dim_in=DIM_IN,
            dim_out=DIM_OUT,
            n_spatial_dims=N_SPATIAL_DIMS,
            spatial_resolution=SPATIAL_RESOLUTION,
            N_layers=3,
            N_res=4,
            N_res_neck=4,
            channel_multiplier=96, # Tuned value
            use_bn=True
        )
    else:
        print(f"Error: Unknown model '{model_name}'")
        sys.exit(1)

    param_count = count_parameters(model)
    print(f"==> Total trainable parameters: {param_count:,}")

if __name__ == "__main__":
    main()
