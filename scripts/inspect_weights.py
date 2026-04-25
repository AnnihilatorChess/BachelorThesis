import torch
import glob
import os

files = glob.glob(r'C:\Users\simon\Documents\GitHub\PDEBench\Model_weights\*.pt')
for f in files:
    print(f"\n--- {os.path.basename(f)} ---")
    try:
        data = torch.load(f, map_location='cpu')
        if isinstance(data, dict):
            if 'model_state_dict' in data:
                state_dict = data['model_state_dict']
            elif 'state_dict' in data:
                state_dict = data['state_dict']
            else:
                state_dict = data
        else:
            state_dict = data.state_dict()

        count = 0
        for k, v in state_dict.items():
            if 'weight' in k or 'weights' in k:
                print(f"{k}: {v.shape}")
                count += 1
                if count >= 15:
                    break
    except Exception as e:
        print(f"Error loading {f}: {e}")
