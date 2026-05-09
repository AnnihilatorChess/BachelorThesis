import time
import torch
from the_well.benchmark.models.cno import CNO

def benchmark_cno(N_layers, N_res, N_res_neck, channel_multiplier):
    model = CNO(
        dim_in=3,
        dim_out=3,
        n_spatial_dims=2,
        spatial_resolution=(128, 128),
        N_layers=N_layers,
        N_res=N_res,
        N_res_neck=N_res_neck,
        channel_multiplier=channel_multiplier
    ).cuda()
    
    x = torch.randn(2, 3, 128, 128).cuda()
    
    # Warmup
    for _ in range(5):
        model(x)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        y = model(x)
        y.sum().backward()
    torch.cuda.synchronize()
    
    print(f"L{N_layers}_R{N_res}_N{N_res_neck}_C{channel_multiplier}: {time.time() - start:.3f} s, Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

if __name__ == "__main__":
    print("Benchmarking CNO configurations...")
    # Baseline PDEBench downsized
    benchmark_cno(3, 4, 4, 32)
    # Reduced N_res
    benchmark_cno(3, 1, 4, 32)
    # Deeper, but smaller N_res
    benchmark_cno(4, 1, 4, 32)
    benchmark_cno(4, 2, 4, 32)
    # Wider, but smaller N_res
    benchmark_cno(3, 1, 4, 48)
    benchmark_cno(4, 1, 4, 48)
