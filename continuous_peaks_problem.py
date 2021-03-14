import numpy as np
import mlrose_hiive as mlrose

from mlrose_hiive import ContinuousPeaksGenerator

random_seed = 42
# Set up the problem

for i in [25, 50, 100, 200]:
    problem = ContinuousPeaksGenerator().generate(seed=random_seed, size=i)
    output_directory = "./continuous_peaks_baseline_"+str(i)

    # Randomized Hill Climbing

    rhc = mlrose.RHCRunner(problem=problem, generate_curves=True,
                           experiment_name="RHC_four_peaks",
                           output_directory=output_directory,
                           seed=random_seed,
                           iteration_list=2 ** np.arange(11),
                           max_attempts=500,
                           restart_list=[25, 75, 100])
    rhc_run_stats, rhc_run_curves = rhc.run()

    # Simulated Annealing

    sa = mlrose.SARunner(problem=problem,
                         experiment_name="SA_four_peaks",
                         output_directory=output_directory,
                         seed=random_seed,
                         iteration_list=2 ** np.arange(11),
                         max_attempts=500,
                         temperature_list=[1, 10, 25, 50, 100, 250, 500, 750, 1000],
                         decay_list=[mlrose.ExpDecay, mlrose.GeomDecay, mlrose.ArithDecay])
    sa_run_stats, sa_run_curves = sa.run()

    # Genetic Algorithm

    ga = mlrose.GARunner(problem=problem,
                         experiment_name="GA_four_peaks",
                         output_directory=output_directory,
                         seed=random_seed,
                         iteration_list=2 ** np.arange(11),
                         max_attempts=500,
                         population_sizes=[100, 200, 300],
                         mutation_rates=[0.1, 0.25, 0.4, 0.55, 0.70])
    ga_run_stats, ga_run_curves = ga.run()

    # MIMIC

    mimic = mlrose.MIMICRunner(problem=problem,
                               experiment_name="MIMIC_four_peaks",
                               output_directory=output_directory,
                               seed=random_seed,
                               iteration_list=2 ** np.arange(11),
                               population_sizes=[100, 200, 300],
                               max_attempts=500,
                               keep_percent_list=[0.25, 0.5, 0.75],
                               use_fast_mimic=True)
    mimic_run_stats, mimic_run_curves = mimic.run()
