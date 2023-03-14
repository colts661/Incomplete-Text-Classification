import sys
import argparse

sys.path.insert(0, 'src')
from models import run_baseline_model, run_final_model


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run script"
    )
    parser.add_argument(
        "target", choices=['test', 'experiment', 'exp'],
        type=str, default='experiment',
        help="run target. Default experiment; if test is selected, run final model on testdata."
    )
    parser.add_argument(
        "-d", "--data", 
        type=str, help="data path",
        default='DBPedia-small'
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=['baseline', 'final'],
        help="model pipeline to run", default='final'
    )

    return parser.parse_args()


if __name__ == "__main__":
    # parse command-line arguments
    args = parse()

    # test target
    if args.target == 'test':
        run_final_model('testdata')

    else:
        # experiment target
        if args.model == 'baseline':
            run_baseline_model(args.data)
        elif args.model == 'final':
            run_final_model(args.data)
