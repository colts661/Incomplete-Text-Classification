import sys
import argparse
from pprint import pprint

sys.path.insert(0, 'src')

from util import *
from data import Data
from baseline_model import *


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run script using runtime configuration defined in `config/`"
    )
    parser.add_argument(
        "target", choices=['test', 'experiment', 'exp'],
        type=str, default='experiment',
        help="run target. Default experiment; if test is selected, ignore all other flags."
    )
    parser.add_argument("-d", "--data", type=str, help="data path", default='nyt/coarse')
    return parser.parse_args()


def test() -> None:
    """
    Run test target
    """
    print('Running Test Data Target:')
    testdata = Data('test', 'testdata')
    testdata.process_corpus(remove_stopwords=False)
    testdata.process_labels(bottom_p=0.5, full_k=1, keep_p=0.6)
    print()
    pprint(testdata.show_statistics())
    print()
    pprint(testdata)

    test_baseline = Baseline_Model(testdata)
    test_pred = test_baseline.run(w2v_config={
        'vector_size': 16,
        'epochs': 2,
        'window': 3,
        'min_count': 1
    })
    test_baseline.evaluate(test_pred)
    

def experiment(dataset: str) -> None:
    print('Running Experiment Target:')
    data = Data('data', dataset)
    data.process_corpus()
    if not hasattr(data, 'labeled_labels'):
        data.process_labels(rerun=True)
    print()
    pprint(data.show_statistics())
    print()
    pprint(data)

    baseline = Baseline_Model(data)
    pred = baseline.run(w2v_config={
        'vector_size': 32,
        'epochs': 5
    })
    baseline.evaluate(pred)


if __name__ == "__main__":
    # parse command-line arguments
    args = parse()

    # test target
    if args.target == 'test':
        test()

    else:
        # experiment target
        if args.target in ['experiment', 'exp']:
            experiment(args.data)
