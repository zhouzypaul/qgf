<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Q-Guided Flow</h1>
      <br>
      <h2><a href="https://arxiv.org/pdf/2606.11087" style="color: #73A4C6">[Paper]</a> &emsp; <a href="https://q-guided-flow.github.io/" style="color: #73A4C6">[Website]</a></h2>
    </summary>
  </ul>
</div>

<img src="assets/teaser.png" width="100%">

</div>

## Overview

Q-guided flow (QGF) is a test-time RL algorithm that trains a reference *flow-matching* policy
via behavioral cloning and a TD-based critic separately, then at inference guides the denoising
process with a novel critic gradient estimator to sample higher-value actions — achieving
policy improvement without any actor-critic training.

## Installation

**With uv (recommended):**

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

**With pip:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

### OGBench (single-task, 100M)

Download the OGBench datasets and set `OGBENCH_DATA_DIR` to the directory containing per-environment
subdirectories of `.npz` slice files:

```
$OGBENCH_DATA_DIR/
  cube-triple-play-100m-v0/    ← *.npz slice files
  cube-quadruple-play-100m-v0/
  puzzle-4x4-play-100m-v0/
  scene-play-100m-v0/
  ...
```

Refer to the [OGBench repository](https://github.com/seohongpark/ogbench) for download instructions.

```bash
export OGBENCH_DATA_DIR=/path/to/ogbench/data
```
The experiment generation scripts below assume this environment variable is set.

## Running a single experiment

All experiments are launched through `main.py`. The agent configuration is passed via
`--agent=agents/<agent>.py` and all agent hyperparameters can be overridden with
`--agent.<key>=<value>`.

### QGF (our method)

```bash
MUJOCO_GL=egl python main.py \
  --agent=agents/qgf.py \
  --agent.denoised_action_approx=one_euler_step_approx \
  --agent.apply_jacobian=False \
  --agent.action_chunking=True \
  --agent.horizon_length=5 \
  --agent.batch_size=1024 \
  --agent.value_network_kwargs.hidden_dims="(1024,1024,1024,1024)" \
  --agent.actor_hidden_dims="(1024,1024,1024,1024)" \
  --agent.discount=0.999 \
  --env_name=cube-triple-play-singletask-task1-v0 \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/cube-triple-play-100m-v0/ \
  --offline_steps=500000 \
  --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12 \
```

The above is an example command for running on `cube-triple-play-singletask-task1-v0`, change the environment name and dataset directory as needed. 
The `--guidance_weights` flag specifies all the guidance weights to try in a for-loop during policy evaluation,
and the evaluation statistics (e.g. success rate) for different guidance weights are saved under different keys.

Checkpoints are saved to `exp/qgf/{wandb_run_group}/{exp_name}/params_{step}.pkl`.


## Launching full paper experiments (via SLURM)

Each `scripts/exp_*.py` generates a SLURM batch script under `sbatch/` and prints commands in debug mode.
The generated scripts use [GNU parallel](https://www.gnu.org/software/parallel/) to run
multiple jobs per GPU — make sure it is installed on your cluster before submitting.


### Train-time methods

These baselines train their own actor/critic jointly — just run the script:

```bash
python scripts/exp_cfgrl.py  && bash sbatch/cfgrl.sh
python scripts/exp_fql.py    && bash sbatch/fql.sh
python scripts/exp_edp.py    && bash sbatch/edp.sh
python scripts/exp_qam.py    && bash sbatch/qam.sh
python scripts/exp_dac.py    && bash sbatch/dac.sh
python scripts/exp_qsm_bc.py && bash sbatch/qsm_bc.sh
```

### Test-time methods

These methods apply guidance at inference time on top of a shared base model (BC as the actor, IQL as the critic).
Train the base model first, then run the desired test-time script:

```bash
# Step 1: train BC+IQL base model
python scripts/bc_iql_train.py && bash sbatch/bc_iql.sh

# Step 2: test-time guidance (set TRAIN_RUN_GROUP to the run_group used above)
TRAIN_RUN_GROUP=bc_iql python scripts/exp_qgf_test_time_eval.py             && bash sbatch/qgf_test_time_eval.sh
TRAIN_RUN_GROUP=bc_iql python scripts/exp_qgf_jacobian_test_time_eval.py    && bash sbatch/qgf_jacobian_test_time_eval.sh
TRAIN_RUN_GROUP=bc_iql python scripts/exp_qfql_test_time_eval.py            && bash sbatch/qfql_test_time_eval.sh
TRAIN_RUN_GROUP=bc_iql python scripts/exp_robust_q.py                       && bash sbatch/robust_q.sh
TRAIN_RUN_GROUP=bc_iql python scripts/exp_grad_step_test_time_eval.py       && bash sbatch/grad_step.sh
```

If you'd like, you can also run evaluation with your chosen test-time method periodically during training. For example, to run just QGF training + evaluation, use:
```bash
python scripts/qgf_train_test.py && bash sbatch/qgf_train_test.sh
```

## Environments
The paper experiments (and scripts above) use the simulated environments from [OGBench](https://github.com/seohongpark/ogbench).
There are also other environment types supported under `envs/`, such as exorl, d4rl, and robomimic, though they are not used in the paper.

## Agents
`agents/` contains the implementation of common RL agents, including QGF, train- and test-time methods compared in the paper, as well as other common RL agents which are not used in the paper.

## Pre-commit hooks

```bash
pre-commit install
```

## Acknowledgments
This codebase is built with reference implementations from [FQL](https://github.com/seohongpark/fql) and [QAM](https://github.com/ColinQiyangLi/qam).

## BibTeX

```bibtex
@article{zhou2026qgf,
  title   = {Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning},
  author  = {Zhou, Zhiyuan and Peng, Andy and Xu, Charles and Li, Qiyang and
             Springenberg, Jost Tobias and Frans, Kevin and Levine, Sergey},
  year    = {2026},
  journal = {arXiv preprint arXiv:2606.11087},
}
```
