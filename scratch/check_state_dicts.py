import torch
import sys

def inspect_checkpoint(filepath):
    print(f"\n--- Inspecting {filepath} ---")
    try:
        ckpt = torch.load(filepath, map_location='cpu', weights_only=True)
        
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # Count conv layers
        conv_layer_indices = set()
        for key in state_dict.keys():
            if key.startswith('convs.'):
                # e.g., 'convs.0.lin_src.weight' -> extract '0'
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    conv_layer_indices.add(int(parts[1]))
                    
        print(f"Number of conv layers detected: {len(conv_layer_indices)}")
        if conv_layer_indices:
            print(f"Layer indices: {sorted(list(conv_layer_indices))}")
    except Exception as e:
        print(f"Failed to load: {e}")

if __name__ == '__main__':
    base_dir = r"d:\projects\cascade sheild\checkpoints"
    inspect_checkpoint(f"{base_dir}\\best_model_3L.pth")
    inspect_checkpoint(f"{base_dir}\\best_model_5L.pth")
    inspect_checkpoint(f"{base_dir}\\best_model_5L_focal.pth")
    inspect_checkpoint(f"{base_dir}\\best_model.pth")
