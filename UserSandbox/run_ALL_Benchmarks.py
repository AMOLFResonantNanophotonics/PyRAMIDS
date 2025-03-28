# -*- coding: utf-8 -*-
"""
Run all the Benchmarks Literature in one push

Inside Benchmark>> Literature folder >> Internal Benchmark
@dpal, fkoenderink

"""
print('!!! RUNNING ALL BENCHMARKS in ONE GO !!!')
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import glob
import importlib.util



def run_all_literature_benchmarks():
    benchmark_files = glob.glob(os.path.join(project_root, "Benchmarks", "Literature","LitBenchmark_*.py"))
    
    if not benchmark_files:
        print("No benchmark files found.")
        return
    
    for benchmark_path in benchmark_files:
        benchmark_name = os.path.basename(benchmark_path)
        print(f"Running {benchmark_name}...")

        spec = importlib.util.spec_from_file_location("benchmark_module", benchmark_path)
        benchmark_module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = benchmark_module
        
        try:
            spec.loader.exec_module(benchmark_module)
        except Exception as e:
            print(f"Error running {benchmark_name}: {e}")


def run_all_internal_benchmarks():
    benchmark_files = glob.glob(os.path.join(project_root, "Benchmarks", "Internal_Consistency","InternalBenchmark_*.py"))
    
    if not benchmark_files:
        print("No benchmark files found.")
        return
    
    for benchmark_path in benchmark_files:
        benchmark_name = os.path.basename(benchmark_path)
        print(f"Running {benchmark_name}...")

        spec = importlib.util.spec_from_file_location("benchmark_module", benchmark_path)
        benchmark_module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = benchmark_module
        
        try:
            spec.loader.exec_module(benchmark_module)
        except Exception as e:
            print(f"Error running {benchmark_name}: {e}")
            
            
if __name__ == "__main__":
    run_all_literature_benchmarks()
    print('*******************### Now Doing INTERNAL CONSISTENCY runs ###**************************')
    run_all_internal_benchmarks()
