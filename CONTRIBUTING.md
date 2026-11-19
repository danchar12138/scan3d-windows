# Contributing

Thanks for helping. Bug reports, benchmark results on other GPUs or capture types, and pull requests are all welcome.

## Reporting a problem

Open an issue and include:

- Windows version, GPU model and driver version
- the output of `tools\python\python.exe app\scan3d.py doctor`
- the template and the stage that failed
- the job's `logs` folder (`runs\<job>\logs`) and `runs\<job>\status.json`

Please do not attach photos you do not want to share. If a bug only shows with a particular capture, describe it
(photo count, resolution, scene type) and, if you can, share a small public dataset that shows it.

## Development setup

```
git clone https://github.com/danchar12138/scan3d-windows.git
cd scan3d-windows
setup.cmd
```

`setup.cmd` downloads the embedded Python, Spirula Studio and COLMAP into `tools\` (git-ignored). The code is plain Python 3.12
(standard library only) and plain browser JavaScript with no build step.

Run the tests with the bundled Python, by script path:

```
tools\python\python.exe app\tests\test_workflow.py
tools\python\python.exe app\tests\test_server.py
```

There is one test file per module under `app\tests`; the server tests use a stub pipeline, so they need no GPU. Run all of them before
you open a pull request.

## Layout

| Path | What |
|---|---|
| `app/workflow.py` | stages, templates, fingerprints, run planning |
| `app/pipeline.py` | runs the planned stages and writes stage records |
| `app/server.py` | HTTP API and static files for the web UI |
| `app/models.py` | model downloads and licence records |
| `app/meshpost.py`, `app/coverage.py`, `app/splatconv.py`, `app/orient.py` | mesh post-processing, capture coverage, splat conversion, orientation |
| `app/web/` | the UI (`app.js` job list, `workflow.js` graph, `jobview.js` and `viewer3d.js` viewers) |
| `app/web/vendor/` | unmodified third-party browser libraries; do not edit, see its README |

## Principles the project follows

- **Explicit settings over automatic rules.** A behaviour change should be a visible switch or parameter on a node, not a hidden heuristic.
- **No device or user-agent sniffing.** The page decides from explicit values (viewport width, user choice).
- **Every output must be viewable in the page**, and every quality-improving option must be reachable from the UI.
- **Back a default with a measurement.** A new default or a new option needs numbers (hold-out PSNR/SSIM/LPIPS with `--set train.eval_holdout=true`, or timings) in the pull request. Options that did not help can still be added, but stay off.
- **No hung processes.** Anything that starts an external program needs a time limit, a no-output watchdog and cancellation.
- **New models.** Adding a model means stating its licence in the node, in the README table and in THIRD_PARTY.md. Say clearly if it is more restrictive than the ones already there.

## Pull requests

- Keep a pull request to one change; describe what you measured.
- Add or update tests for behaviour you change in `app\workflow.py`, `app\server.py` or the pipeline.
- Do not commit `tools\`, `data\`, `runs\` or `user-settings.json` (they are git-ignored).
- By contributing you agree your work is released under the MIT license of this repository.
