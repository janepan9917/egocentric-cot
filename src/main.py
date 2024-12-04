from prompt import *
from model import *
import yaml

def _main(args):
    # 0. Set up some utilities
    np.random.seed(0)
    model_config = {
        "model_name": args.model_name,
        "api_key": args.api_key
    }
    model = get_model(model_config)

    if not os.path.exists(args.output_dir):
        output_dir. 
    # 1. Load prompts
    with open(args.prompt_fn, "r") as file:
        prompt_config = yaml.safe_load(file)

    direct_prompt = get_prompts(
        config=prompt_config,
        prompt_type="direct",
    )

    cot_prompt = get_prompts(
        config=prompt_config,
        prompt_type="cot",
    )

    
    direct_output = model.generate(direct_prompt)
    cot_output = model.generate(cot_prompt)

    import pdb; pdb.set_trace()

    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str,
                        help="Path to config file")
    parser.add_argument("--prompt_fn", type=str, default="../configs/interaction.yaml",
                        help="Path to prompt config files")

    # Model arguments
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name of model")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Number of examples.")
    
    parser.add_argument('-v', '--overwrite_old_output',
                    action='store_true')


    args = parser.parse_args()
    return _main(args)

if __name__ == "__main__":
    main()