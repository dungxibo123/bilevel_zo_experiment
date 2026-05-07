
python main.py --dim 1 --alg3_T 200 --alg2_K 80 --alg3_eta 0.03 --alg2_mu 0.06 --seed_outer 120 --rho 0.05 --epsilon_g 0.25 --log_path logs/dims/run_001.csv --plot_dir plots/dims/1/ --plot
python main.py --dim 2 --alg3_T 200 --alg2_K 80 --alg3_eta 0.03 --alg2_mu 0.06 --seed_outer 220 --rho 0.05 --epsilon_g 0.25 --log_path logs/dims/run_002.csv --plot_dir plots/dims/2/ --plot
python main.py --dim 5 --alg3_T 200 --alg2_K 80 --alg3_eta 0.03 --alg2_mu 0.06 --seed_outer 520 --rho 0.05 --epsilon_g 0.25 --log_path logs/dims/run_005.csv --plot_dir plots/dims/5/ --plot

python main.py --dim 10 --alg3_T 200 --alg2_K 80 --alg3_eta 0.03 --alg2_mu 0.06 --seed_outer 1020 --rho 0.05 --epsilon_g 0.25 --log_path logs/dims/run_0010.csv --plot_dir plots/dims/10/ --plot
python main.py --dim 50 --alg3_T 200 --alg2_K 80 --alg3_eta 0.03 --alg2_mu 0.06 --seed_outer 5020 --rho 0.05 --epsilon_g 0.25 --log_path logs/dims/run_0050.csv --plot_dir plots/dims/50/ --plot
python main.py --dim 100 --alg3_T 200 --alg2_K 80 --alg3_eta 0.03 --alg2_mu 0.06 --seed_outer 10020 --rho 0.05 --epsilon_g 0.25 --log_path logs/dims/run_00100.csv --plot_dir plots/dims/100/ --plot
