import multiprocessing
import os
from datetime import datetime
from typing import Optional

import numpy as np
import threadpoolctl

from plainmp.problem import Problem
from plainmp.trajectory import Trajectory


class ParallelSolver:
    def __init__(self, solver, n_process: int = 4):
        self.internal_solver = solver
        self.n_process = n_process

    def _parallel_solve_inner(self, problem: Problem, guess: Optional[Trajectory] = None):
        """assume to be used in multi processing"""
        # prevend numpy from using multi-thread
        unique_seed = datetime.now().microsecond + os.getpid()
        np.random.seed(unique_seed)
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            return self.internal_solver.solve(problem, guess)

    def solve(self, problem: Problem, guess: Optional[Trajectory] = None):
        processes = []
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        for i in range(self.n_process):
            p = multiprocessing.Process(
                target=lambda: result_queue.put(self._parallel_solve_inner(problem, guess))
            )
            processes.append(p)
            p.start()

        for _ in range(self.n_process):
            result = result_queue.get()
            if result.traj is not None:
                for p in processes:
                    p.terminate()
                for p in processes:
                    p.join()
                return result
        return result.abonormal()
