import torch

# simulate an FNO forward pass with autocast
x = torch.randn(2, 3, 128, 384, dtype=torch.float32)

with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
    # a simple bypass
    bypass = torch.nn.Conv2d(3, 3, 1)
    x_skip = bypass(x) # this will be bfloat16
    
    # simulate patched FFT
    x_fft_input = x_skip.float() # patched rfftn converts bfloat16 to float32
    
    # an FFT
    x_fft = torch.fft.rfftn(x_fft_input, dim=(-2, -1))
    
    # some operation in spectral domain
    x_fft = x_fft * 2.0
    
    # inverse FFT
    x_ifft = torch.fft.irfftn(x_fft, s=(128, 384), dim=(-2, -1))
    
    # add bypass
    out = x_skip + x_ifft
    print("x_skip dtype:", x_skip.dtype)
    print("x_ifft dtype:", x_ifft.dtype)
    print("out dtype:", out.dtype)
