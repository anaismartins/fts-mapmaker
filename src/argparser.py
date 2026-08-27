import argparse

# set up line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity.")
parser.add_argument("--plots", type=str, default="none", help="Plot specific plots depending on the" 
                    "type of run. Default is 'none', which means no plots. Options are: 'debug', "
                    "'paper_only'.")
parser.add_argument("--nworkers", type=int, default=None, help="Override the number of worker "
                    "processes used for scanning batches.")
parser.add_argument("--run-name", type=str, default="profiling.txt", help="Name of the run for "
                    "profiling output.")
parser.add_argument("--noise", action="store_false", help="Add no noise to the simulated IFGs.")
parser.add_argument("--sim-type", type=str, help="Type of simulation to run." \
                    "Options are: 'fossil', 'firas'. One of the options must be specified.")
parser.add_argument("--cg-dummy", action="store_true",
                    help="Run the comparison using a dummy CG map.")
parser.add_argument("--firas_ss", action="store_false",
                    help="Use the FIRAS scanning strategy instead of the FOSSIL one on the FIRAS "
                    "sims.")
parser.add_argument("--chunk-size", type=int, default=200_000, help="Number of recorded IFGs "
                    "(rows) processed at a time when building the FIRAS simulation. Lower this if "
                    "the process runs out of memory / starts swapping, e.g. when using the FOSSIL "
                    "scanning strategy which has far more recorded IFGs than FIRAS.")
args = parser.parse_args()