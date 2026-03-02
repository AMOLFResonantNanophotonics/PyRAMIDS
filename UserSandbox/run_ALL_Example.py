"""
UserSandbox batch runner for example scripts.

Runs:
- Example/RoutineUse_Examples/Example_*.py
- Example/PaperReproduce_Examples/Fig*.py

Each script is executed from its own folder so relative outputs
(e.g., pdfimages/) are written locally to that example folder.
"""
print("### UserSandbox Runner: All Examples (Routine Use + Paper Reproduction) ###")
#%%
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


#%%

import glob
import importlib.util


def run_all_examples():
    benchmark_files = glob.glob(os.path.join(project_root, "Example", "RoutineUse_Examples","Example_*.py"))
    
    if not benchmark_files:
        print("No benchmark files found.")
        return
    
    for benchmark_path in benchmark_files:
        benchmark_name = os.path.basename(benchmark_path)
        print(f"Running {benchmark_name}...")

        spec = importlib.util.spec_from_file_location("benchmark_module", benchmark_path)
        benchmark_module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = benchmark_module
        
        prev_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(benchmark_path))
            spec.loader.exec_module(benchmark_module)
        except Exception as e:
            print(f"Error running {benchmark_name}: {e}")
        finally:
            os.chdir(prev_cwd)


def run_all_reproducePaperPanels():
    benchmark_files = glob.glob(os.path.join(project_root, "Example", "PaperReproduce_Examples","Fig*.py"))
    
    if not benchmark_files:
        print("No benchmark files found.")
        return
    
    for benchmark_path in benchmark_files:
        benchmark_name = os.path.basename(benchmark_path)
        print(f"Running {benchmark_name}...")

        spec = importlib.util.spec_from_file_location("benchmark_module", benchmark_path)
        benchmark_module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = benchmark_module
        
        prev_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(benchmark_path))
            spec.loader.exec_module(benchmark_module)
        except Exception as e:
            print(f"Error running {benchmark_name}: {e}")
        finally:
            os.chdir(prev_cwd)
            
            
if __name__ == "__main__":
    run_all_examples()
    print('*******************### Now Doing Paper Figure runs ###**************************')
    run_all_reproducePaperPanels()
    print("Paper figures are saved as PDFs in each example script folder (pdfimages/).")
