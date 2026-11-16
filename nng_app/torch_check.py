"""Runs the torch.nn code exported by the viewer and reports whether it works."""
import os
import pathlib
import subprocess
import sys
import tempfile

CHECK_TIMEOUT_SECONDS = 300


def run_generated_model_check(model_code):
    """Execute exported model code in a separate Python process.

    The exported file ends with a self-check that builds the model, feeds random inputs of
    shape (B=4, *input_dims), asserts every output shape, back-propagates a dummy loss and
    asserts every parameter and input received a finite gradient.

    Args:
        model_code (str): full text of the exported model.py.

    Returns:
        dict: {
            "passed" (bool): True when the script exited cleanly and printed its OK line,
            "output" (str): combined stdout and stderr of the run,
        }
    """
    child_environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="nng_check_") as work_dir:
        script_path = pathlib.Path(work_dir) / "model.py"
        script_path.write_text(model_code, encoding="utf-8")
        try:
            completed_run = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CHECK_TIMEOUT_SECONDS,
                env=child_environment,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "output": f"Timed out after {CHECK_TIMEOUT_SECONDS} s"}

    run_output = (completed_run.stdout or "") + (completed_run.stderr or "")
    passed = completed_run.returncode == 0 and "\nOK:" in run_output
    return {"passed": passed, "output": run_output}
