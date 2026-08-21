from solver import Solver
from glob import glob
import os


def main():
    solver = Solver()
    solver.train()
    solver.valid(show_evaluations=True)

    # run_dir = r'result\sea depth\L4\20260615_222122'
    # solver = Solver(run_dir=run_dir)
    solver.test(2021, 1, 1)
    solver.test(2022, 5, 1)
    solver.test(2023, 9, 1)


def valid_all():
    run_dirs = glob(os.path.join(r'result\lr selection', '**', '????????_??????'), recursive=True)
    for run_dir in run_dirs:
        solver = Solver(run_dir=run_dir)
        solver.valid(show_evaluations=True)


if __name__ == '__main__':
    main()
