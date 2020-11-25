#/bin/sh
#
# Run from rootpath with: source ./tests/unit_analysis_all.sh
#
#

# Bernoulli simulation
python ./covidgen/sim.py --R 10000
python ./analysis/analyze_sim.py

# IFR computation and analysis
python ./analysis/run_compute_IFR.py
python ./analysis/run_analyze_IFR.py

# Kernel generation
python ./analysis/run_generate_kernels.py
python ./analysis/run_plot_kernels.py
python ./analysis/run_visualize_deconvolution.py

# Confidence/credible intervals
python ./analysis/run_bayesian_2D.py
python ./analysis/run_bayesian_ratio.py
python ./analysis/run_bootstrap_comp.py
python ./analysis/run_neyman_belt.py
python ./analysis/run_profile_likelihood.py
python ./analysis/run_running_counts.py

# Time-series visualization
python ./tests/run_visualize_OWID.py

# These are slow
python ./analysis/run_coverage_sim.py
python ./analysis/run_simloop.py
python ./analysis/run_mc_confidence.py

