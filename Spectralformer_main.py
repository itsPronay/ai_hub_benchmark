import argparse
import qai_hub as hub
from utils.initialize_model import initializeResnetModel
from utils.benchmark import run_compile, run_profile, get_traced_model
from utils.extract_metrices import extract_metrics_from_profile, log_top15_table, log_op_type_table
from spectralFormer.vit import ViT
import wandb
import os

parser = argparse.ArgumentParser(description='Spectralformer-benchmark-f')
parser.add_argument('--patches', type=int, default=7)
parser.add_argument('--band_patches', choices=[1, 3, 5, 7, 9, 11], type=int, default=1)
parser.add_argument('--mode', choices=['CAF', 'ViT'], default='ViT')
parser.add_argument('--num_classes', type=int, default=9, help='Output class of the model')
parser.add_argument('--band', type=int, default=100)

parser.add_argument('--ai_hub_device',
                    choices=[
                        'Dragonwing IQ-9075 EVK', 'Google Pixel 10 Pro XL','QCS8550 (Proxy)',
                        'Samsung Galaxy S24 (Family)','Samsung Galaxy S24 Ultra',
                        'Google Pixel 3', 'Google Pixel 6'
                    ],
                    default='Samsung Galaxy S24 (Family)',
                    help='Device to run on ai hub')

parser.add_argument("--wandb_project", default="SpectralFormer")
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
args, _ = parser.parse_known_args()

def patchDim(patch, band_patch):
    return patch ** 2 * band_patch

def main():
    model = ViT(
        image_size = args.patches,
        near_band = args.band_patches,
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

    input_shape: tuple[int, ...] = (1, args.band, patchDim(args.patches, args.band_patches))
    traced_model = get_traced_model(input_shape, model)

    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = "{0}_{1}_{2}_{3}".format(
                args.ai_hub_device, 
                args.mode,
                args.band,
                patchDim(args.patches, args.band_patches)
            ),
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
      'patch' : args.patches,
      'band_patch' : args.band_patches,
      'band' : args.band,
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
