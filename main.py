import argparse
from utils.initialize_model import initializeResnetModel
from utils.benchmark import run_compile, run_profile, get_traced_model
from utils.extract_metrices import extract_metrics_from_profile, log_top15_table, log_op_type_table
import wandb
import os
import qai_hub as hub

parser = argparse.ArgumentParser(description='resnet-benchmark')
parser.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18', help='model to benchmark')
parser.add_argument('--image_size', type=int, choices=[112, 224, 336, 448, 560, 672, 784, 896, 1008, 1120], default=112)
parser.add_argument('--num_class', type=int, default=2, help='Output class of the model')
parser.add_argument('--ai_hub_device',
                    choices=['Dragonwing IQ-9075 EVK', 'Google Pixel 10 Pro XL','QCS8550 (Proxy)','Samsung Galaxy S24 (Family)','Samsung Galaxy S24 Ultra'],
                    default='Samsung Galaxy S24 (Family)',
                    help='Device to run on ai hub')
parser.add_argument("--wandb_project", default="resnet")
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
parser.add_argument("--log_top_15_op", type=bool, default=True)
parser.add_argument("--log_op_type_table", type=bool, default=True)
args, _ = parser.parse_known_args()

def main():
    model = initializeResnetModel(args.model, args.num_class)
    device = hub.Device(args.ai_hub_device)

    input_shape: tuple[int, ...] = (1, 3, args.image_size, args.image_size)
    traced_model = get_traced_model(input_shape, model)

    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = args.ai_hub_device + "_" + str(args.image_size),
            mode = args.wandb_mode,
            config = vars(args)
        )
    
    compiled_model = run_compile(traced_model, device, input_shape)

    profiled_model = run_profile(compiled_model, device)

    # if you want to download profile job result in your device , keep this line
    profiled_model.download_results(os.path.join(os.getcwd(), 'job_result')) 

    profile = profiled_model.download_profile()

    metrices = {
      'model_name' : args.model,
      'device' : str(args.ai_hub_device),
      'image_size' : str(args.image_size),
      **extract_metrics_from_profile(profile)
    }

    if args.wandb_mode != 'disabled':
        wandb.log(metrices)
        if args.log_top_15_op == True:
            log_top15_table(profile)
        if args.log_op_type_table == True:
            log_op_type_table(profile)
        wandb.finish()

if __name__ == '__main__':
    main()
