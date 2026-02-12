import pytest
import shutil

from src.pyvawt.main import run_simulation, load_config
from src.pyvawt.utils import read_dat, save_config

# How to run this test: 'uv run pytest'

def test_simulation(tmp_path):
    '''
    Test the accuracy of the Vertical-Axis Wind Turbine (VAWT) simulation.

    This test temporarily replaces the default configuration file 
    (``src/pyvawt/config/config.yaml``) with a test configuration 
    (``test/data/config.yaml``), runs the simulation, and compares 
    the generated results against a reference dataset. After the test, 
    the original configuration is restored, regardless of whether 
    the test passes or fails.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory fixture provided by pytest (not explicitly used 
        in this implementation, but available for handling temporary files 
        if needed).

    Raises
    ------
    AssertionError
        If any value in the output simulation results differs from the 
        reference dataset by more than the specified tolerance 
        (``rel=1e-6``).

    Notes
    -----
    - The test ensures reproducibility of simulation results by 
      directly comparing numerical outputs with a precomputed 
      reference dataset.
    - A backup of the original configuration is created and restored 
      using ``shutil.copyfile`` and ``shutil.move`` to avoid 
      permanent modification of project files.

    See Also
    --------
    run_simulation : Runs a batch of VAWT aerodynamic simulations.
    read_dat : Reads simulation results from a ``.dat`` file into a 
        list of lists of floats.
    '''
    reference_naca0018_file = 'test/data/reference_results_naca0018.dat'
    output_naca0018_file = 'src/results/temporary_results/naca0018_ch1p75_sol0p1_vinf6p23_delta0p0_r35p0/results_naca0018.dat'
    test_config_path = 'test/data/config.yaml'
    original_config_path = 'src/pyvawt/config/config.yaml'
    backup_path = original_config_path + ".bak"

    shutil.copyfile(original_config_path, backup_path)

    try:
        shutil.copyfile(test_config_path, original_config_path)
        run_simulation()

        reference_data = read_dat(reference_naca0018_file)
        output_data = read_dat(output_naca0018_file)

        
        for line_index, (ref_row, out_row) in enumerate(zip(reference_data, output_data)):
            for col_index, (ref_val, out_val) in enumerate(zip(ref_row, out_row)):
                # pytest.approx handles small floating-point differences
                assert out_val == pytest.approx(
                    ref_val, rel=1e-6
                ), f'Mismatch at row {line_index}, column {col_index}'

    finally:
        shutil.move(backup_path, original_config_path)
