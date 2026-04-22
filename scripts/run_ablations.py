
import subprocess
import os

base_dir = '/home/barshikar.s/depth-aware-dehazing'

ablations = [
    {
        'name': 'depth_attn_early',
        'script': 'train_depth_attention.py',
        'args': '--injection early --epochs 30 --experiment_dir experiments/ablation_attn_early'
    },
    {
        'name': 'depth_attn_late',
        'script': 'train_depth_attention.py',
        'args': '--injection late --epochs 30 --experiment_dir experiments/ablation_attn_late'
    },
    {
        'name': 'depth_attn_direct',
        'script': 'train_depth_attention.py',
        'args': '--attention_type direct --epochs 30 --experiment_dir experiments/ablation_attn_direct'
    },
    {
        'name': 'joint_lambda_0.1',
        'script': 'train_depth_joint.py',
        'args': '--lambda_depth 0.1 --epochs 30 --batch_size 4 --experiment_dir experiments/ablation_joint_lambda01'
    },
    {
        'name': 'joint_lambda_1.0',
        'script': 'train_depth_joint.py',
        'args': '--lambda_depth 1.0 --epochs 30 --batch_size 4 --experiment_dir experiments/ablation_joint_lambda10'
    },
]

print("Ablation studies")

for i, abl in enumerate(ablations):
    print(f"\n[{i+1}/{len(ablations)}] Running: {abl['name']}")
    
    cmd = f"cd {base_dir} && python scripts/{abl['script']} {abl['args']}"
    print(f"Command: {cmd}\n")

print("To run ablations, uncomment subprocess.run() in the script")
print("Or run each command manually")