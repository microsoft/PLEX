import torch
max_ep_len = 0

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODALITIES = ['image', 'depth', 'proprio', 'action', 'reward']

full_state_mode = False

def is_full_state(camera_names):
    if 'FULL_STATE' in camera_names:
        assert len(camera_names) == 1, "If FULL_STATE is present among camera names, it must be the only camera name."
        return True
    else:
        return False