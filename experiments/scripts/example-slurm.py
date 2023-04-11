"""
An example submitit script for running multiple jobs in parallel on a SLURM cluster.
"""
import datetime
import shlex
import sys
import time
from pathlib import Path

try:
    import submitit
    from tap import Tap
except Exception as e:
    print('Please install submitit with `pip install submitit`')
    print('Please install tap with `pip install typed-argument-parser`')
    raise(e)


class SubmitItArgs(Tap):
    partition: str = 'your-slurm-partition'
    submit: bool = False


# Args
args = SubmitItArgs().parse_args()

# Commands
commands = []
for category in ['hydrant', 'toytruck']:
    command = f"""python main.py dataset.category={category} dataloader.batch_size=24 dataloader.num_workers=8 run.vis_before_training=True run.val_before_training=True run.name=check_train__{category}__ebs_24"""
    commands.append(command)

# Create executor
Path("slurm_logs").mkdir(exist_ok=True)
executor = submitit.AutoExecutor(folder="slurm_logs")
executor.update_parameters(
    tasks_per_node=1,
    timeout_min=36 * 60,
    slurm_partition=args.partition,
    slurm_gres="gpu:1",
    slurm_constraint="volta32gb",
    slurm_job_name="submititjob",
    cpus_per_task=8,
    mem_gb=32.0,
)

# Check
if not args.submit:
    for command in commands:
        print(command)
    sys.exit()

# Submitit via SLURM array
print('Start')
print(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
start_time = time.time()
jobs = []
with executor.batch():
    print('SUBMITTING:')
    for command in commands:
        function = submitit.helpers.CommandFunction(shlex.split(command), verbose=True)
        job = executor.submit(function)
        jobs.append(job)

# Then wait until all jobs are completed:
outputs = [job.result() for job in jobs]
print(f'Finished all ({len(outputs)}) jobs in {time.time() - start_time:.1f} seconds')
print(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))