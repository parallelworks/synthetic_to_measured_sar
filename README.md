# Synthetic to Measured SAR
The code in script `main_experiment41_tester.py` from repository [inkawhich/synthetic-to-measured-sar](https://github.com/inkawhich/synthetic-to-measured-sar) was modified to run every iteration in parallel using [Parsl](https://parsl-project.org/parslfest2022.html) in a SLURM cluster.

The default parameters in the input form correspond to the default parameters in the repository and the following articles:

[1] - Nathan Inkawhich, Matthew Inkawhich, Eric Davis, Uttam Majumder, Erin Tripp, Chris Capraro and Yiran Chen, "Bridging a Gap in SAR-ATR: Training on Fully Synthetic and Testing on Measured Data," Preprint (Under Review), 2020.

[2] - Benjamin Lewis, Theresa Scarnati, Elizabeth Sudkamp, John Nehrbass, Stephen Rosencrantz, Edmund Zelnio, "A SAR dataset for ATR development: the Synthetic and Measured Paired Labeled Experiment (SAMPLE)," Proc. SPIE 10987, Algorithms for Synthetic Aperture Radar Imagery XXVI, 109870H (14 May 2019); https://doi.org/10.1117/12.2523460

![synthetic_workflow_dag](https://user-images.githubusercontent.com/28575746/224402560-c0d594a9-40bf-466c-8495-a0ea8d09e379.png)
