import argparse
import qai_hub as hub
from utils.initialize_model import initializeResnetModel
from utils.benchmark import run_compile, run_profile, get_traced_model
from utils.extract_metrices import extract_metrics_from_profile, log_top15_table, log_op_type_table
from model.vit import ViT
from model.vit_batch_processing import ViTbatchProcessing
import wandb
import os
import torch
import torch.nn as nn
import time

parser = argparse.ArgumentParser(description='ViT-benchmark-on-pavia')
parser.add_argument('--patches', type=int, default=15, help='Spatial patch size (height and width)')
parser.add_argument('--mode', choices=['CAF', 'ViT'], default='ViT')
parser.add_argument('--num_classes', type=int, default=9, help='Output class of the model')
parser.add_argument('--band', type=int, default=100, help='Number of spectral bands')
parser.add_argument('--total_samples', type=int, default=1, help='Total number of samples in full dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for internal processing')

parser.add_argument('--ai_hub_device',
                    choices=[
                        'Dragonwing IQ-9075 EVK', 'Google Pixel 10 Pro XL','QCS8550 (Proxy)',
                        'Samsung Galaxy S24 (Family)','Samsung Galaxy S24 Ultra',
                        'Google Pixel 3', 'Google Pixel 6', 'Snapdragon X2 Elite CRD',
                        'Snapdragon X Elite CRD', 'Snapdragon 8 Elite Gen 5 QRD'
                    ],
                    default='Samsung Galaxy S24 (Family)',
                    help='Device to run on ai hub')

parser.add_argument("--wandb_project", default="ViT-benchmark-on-pavia", help="WandB project name")
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
args, _ = parser.parse_known_args()

def main():
    # Use ViTbatchProcessing for batch-wise processing
    model = ViTbatchProcessing(
        image_size=args.patches,
        near_band=1,  
        num_patches=args.band,
        num_classes=args.num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode=args.mode,
        total_samples=args.total_samples,
        batch_size=args.batch_size
    )
    model = model.to("cpu").eval()

    # 5D input for batch processing: (total_samples, 1, bands, height, width)
    # This matches real training data format: torch.Size([42776, 1, 30, 15, 15])
    input_shape: tuple[int, ...] = (args.total_samples, 1, args.band, args.patches, args.patches)
    
    print(f"Using ViTbatchProcessing:")
    print(f"  Input shape: {input_shape}")
    print(f"  Total samples: {args.total_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches: {(args.total_samples + args.batch_size - 1) // args.batch_size}")
    device = hub.Device(args.ai_hub_device)
    
    traced_model = get_traced_model(input_shape, model)

    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = f"{args.ai_hub_device}_band:{args.band}_patch:{args.patches}_Samples:{args.total_samples}",
            mode = args.wandb_mode,
            config = vars(args)
        )

    compiled_model = run_compile(traced_model, device, input_shape)

    profiled_model = run_profile(compiled_model, device)

    # if you want to download profile job result in your device , keep this line
    profiled_model.download_results(os.path.join(os.getcwd(), 'SpectralFormer result')) 

    profile = profiled_model.download_profile()

    metrices = {
      'model_name' : 'SpectralFormer',
      'device' : str(args.ai_hub_device),
      'mode' : args.mode,
      'spatial_patch_size' : args.patches,
      'spectral_bands' : args.band,
      'num_classes' : args.num_classes,
      'total_samples' : args.total_samples,
      'batch_size' : args.batch_size,
      'num_batches' : (args.total_samples + args.batch_size - 1) // args.batch_size,
      'input_shape' : str(input_shape),
      **extract_metrics_from_profile(profile)
    }

    if args.wandb_mode != 'disabled':
        wandb.log(metrices)

        wandb.log({"log_op_type_table": log_op_type_table(profile)})
        wandb.log({"log_top15_table": log_top15_table(profile)})

        wandb.finish()

if __name__ == '__main__':
    main()
