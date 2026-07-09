# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: eyewire2-functional-analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run all tutorial scripts
#
# Smoke-runs every script under `scripts/tutorial/` (except this one) as a
# subprocess, with a non-interactive Matplotlib backend so `plt.show()` calls
# don't block. Each script runs with its own directory as the working
# directory, same as when opened and run interactively in Jupyter. Useful to
# sanity check the tutorials after changing the `eyewire2_functional_analysis`
# library or the data layout.

# %%
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THIS_FILE = os.path.abspath(__file__)


# %%
def find_tutorial_scripts():
    scripts = []
    for root, _dirs, files in os.walk(HERE):
        for name in files:
            if not name.endswith('.py'):
                continue
            path = os.path.abspath(os.path.join(root, name))
            if path == THIS_FILE:
                continue
            scripts.append(path)
    return sorted(scripts)


# %%
env = os.environ.copy()
env['MPLBACKEND'] = 'Agg'
# plt.show() is correct in the scripts (they're meant to run interactively in
# Jupyter too) but is a no-op under Agg and warns every time; silence just that.
env['PYTHONWARNINGS'] = 'ignore:FigureCanvasAgg is non-interactive:UserWarning'

results = []
for script in find_tutorial_scripts():
    rel = os.path.relpath(script, HERE)
    print(f"\n=== Running {rel} ===")
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(script), env=env)
    results.append((rel, result.returncode))

# %% [markdown]
# ## Summary

# %%
print("\n=== Summary ===")
for rel, code in results:
    status = "OK" if code == 0 else f"FAILED (exit {code})"
    print(f"{status:20s} {rel}")

failed = [rel for rel, code in results if code != 0]
if failed:
    print(f"\n{len(failed)} / {len(results)} tutorial script(s) failed")
    sys.exit(1)
else:
    print(f"\nAll {len(results)} tutorial script(s) ran successfully")
