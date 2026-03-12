import argparse
import qai_hub as hub
from utils.benchmark import run_compile, run_profile, get_traced_model
from utils.extract_metrices import extract_metrics_from_profile, log_top15_table, log_op_type_table
from model.vit import ViT
import wandb
import os
import torch
import torch.nn as nn
import time

parser = argparse.ArgumentParser(description='Edge-benchmark-ViT')
parser.add_argument('--patches', type=int, default=15, help='Spatial patch size (height and width)')
parser.add_argument('--mode', choices=['CAF', 'ViT'], default='ViT')
parser.add_argument('--num_classes', type=int, default=9, help='Output class of the model')
parser.add_argument('--band', type=int, default=30, help='Number of spectral bands')

parser.add_argument('--ai_hub_device',
                    choices=[
                        'Dragonwing IQ-9075 EVK', 'Google Pixel 10 Pro XL','QCS8550 (Proxy)',
                        'Samsung Galaxy S24 (Family)','Samsung Galaxy S24 Ultra',
                        'Google Pixel 3', 'Google Pixel 6', 'Snapdragon X2 Elite CRD',
                        'Snapdragon X Elite CRD', 'Snapdragon 8 Elite Gen 5 QRD'
                    ],
                    default='Samsung Galaxy S24 (Family)',
                    help='Device to run on ai hub')

parser.add_argument("--wandb_project", default="Edge-benchmark_of_vit", help="WandB project name")
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
args, _ = parser.parse_known_args()

def main():

    input_shape = [
        100, # 10 x 10
        250, # 25 x 10
        500, # 50 x 10
        1000, # 100 x 10
        # 1500, # 150 x 10
        2000, # 200 x 10
        4000, # 400 x 10
        8000, # 800 x 10
        16000 # 1600 x 10
    ]
    
    # Use ViTbatchProcessing for batch-wise processing
    model = ViT(
        image_size = args.patches,
        near_band = 1,
        num_patches = args.band,
        num_classes = args.num_classes,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = args.mode
    )

    model = model.to("cpu").eval()

    device = hub.Device(args.ai_hub_device)

    for i in range(len(input_shape)):

        shape = (input_shape[i], args.band, args.patches * args.patches * 1) 
        print(f"Testing with input shape: {shape} (H*W, C, patch_dim)")

        if args.wandb_mode != 'disabled':
            wandb.init(
                project = args.wandb_project,
                name = f"Input shape: {shape}, Device: {args.ai_hub_device}, mode: {args.mode}",
                mode = args.wandb_mode,
                config = vars(args)
            )
        
        current_shape: tuple[int, ...] = (input_shape[i], args.band, args.patches * args.patches * 1)  # (H*W, C, patch_dim)
    
        traced_model = get_traced_model(current_shape, model)

        compiled_model = run_compile(traced_model, device, current_shape)

        profiled_model = run_profile(compiled_model, device)
        profile = profiled_model.download_profile()

        # if you want to download profile job result in your device , keep this line
        # profiled_model.download_results(os.path.join(os.getcwd(), 'SpectralFormer result')) 

        metrices = {
            'input_shape' : str(current_shape),
            'device' : str(args.ai_hub_device),
            'mode' : args.mode,
            'spatial_patch_size' : args.patches,
            'spectral_bands' : args.band,
            'num_classes' : args.num_classes,
            **extract_metrics_from_profile(profile)
        }

        if args.wandb_mode != 'disabled':
            wandb.log(metrices)

            wandb.log({"log_op_type_table": log_op_type_table(profile)})
            wandb.log({"log_top15_table": log_top15_table(profile)})

            wandb.finish()

if __name__ == '__main__':
    main()
