import argparse

# set up line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity.")
parser.add_argument("--plots", type=str, default="none", help="Plot specific plots depending on the" 
                    "type of run. Default is 'none', which means no plots. Options are: 'debug', "
                    "'paper_only'.")
parser.add_argument("--workers", type=int, default=None, help="Override the number of worker "
                    "processes used for scanning batches.")
parser.add_argument("--run-name", type=str, default="profiling.txt", help="Name of the run for "
                    "profiling output.")
args = parser.parse_args()