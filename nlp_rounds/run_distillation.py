import os
import time
import subprocess
import shlex
from utils import get_metadata
from datetime import datetime
import json

gpu_list = [3, 4, 5, 6]
gpus_per_command = 1
polling_delay_seconds = 15 # Let the time for the model to be loaded
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
free_memory_required = 16000 # in MB # One run on Resnet34 uses 4500M
max_per_gpu = 1 # max number of things per gpu
# gpus_to_leave = {0} # Let this ones free
round_number = 6
if round_number == 5:
    with open("/scratch/jordan.lekeufack/nlp_models/round_5_corrupted_models.txt", "r") as fp:
        corrupted_models = json.load(fp)

path_to_folder = f'/scratch/jordan.lekeufack/nlp_models/distillation/round{round_number}/'
print(f'Round {round_number}, saving at {path_to_folder}')

# Working on Small Resnets first
# architectures = ['resnet18', 'resnet34'] #'denset121' is not working
METADATA = get_metadata(round_number=round_number)
models = list(METADATA.index)
# models_with_data = os.listdir(f'/scratch/data/TrojAI/round{round_number}/supl_data')
# models = [model for model in models if model in models_with_data]
if round_number == 5:
    models = [model for model in models if model not in corrupted_models]

beta = 1
if not os.path.isdir(f'{path_to_folder}/logs/'):
    os.makedirs(f'{path_to_folder}/logs/')

commands_to_run = [('python3 -u -Wignore distillation.py ' + 
                   f'--model_id {model_id} '+
                   f'--round_number {round_number} ' +
                   f'--path_to_folder {path_to_folder} ' +
                   f'--beta {beta} ' + 
                   f'--device DEVICE ', 
                   f'{path_to_folder}/logs/{model_id}.txt')
                  for model_id in models]

commands_to_run.reverse()

def poll_process(process):
    time.sleep(polling_delay_seconds)
    return process.poll()

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free',
         '--format=csv,nounits,noheader'], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

pid_to_process_and_gpus = {}
#  = set(gpu_list)
start_time = datetime.now()
prev_message_time = start_time # 
delay_between_messages_seconds = 10 * 60
process_per_gpu = {g: 0 for g in gpu_list}
print(f'Starting... {start_time:%Y/%m/%d-%H:%M:%S}, Commands to run : {len(commands_to_run)}')

#while len(commands_to_run) > 0 or len(pid_to_process_and_gpus) > 0:
while len(commands_to_run) > 0 or sum(process_per_gpu.values()) > 0:
    time.sleep(polling_delay_seconds)
    # gpus_memory = get_gpu_memory_map()
    # free_gpus = {gpu for gpu in gpus_memory if (gpus_memory[gpu] >= free_memory_required) and not(gpu in gpus_to_leave)}
    free_gpus = {gpu for gpu in process_per_gpu if process_per_gpu[gpu]<max_per_gpu} # Max of two processes per gpu
    now = datetime.now()
    if (now - prev_message_time).seconds >  delay_between_messages_seconds:
        print(f'{now:%Y/%m/%d-%H:%M:%S}, left to run : {len(commands_to_run)}, running : {len(pid_to_process_and_gpus)}')
        prev_message_time = now

    if len(free_gpus) >= gpus_per_command: # Need to let ~30sec for each model to load data using cpu
        gpu = free_gpus.pop()
        if len(commands_to_run) > 0:
            command, log_path = commands_to_run.pop()
        else:
            break
        command = command.replace('DEVICE', str(gpu)) # Specific to one gpu per command.
        # free_gpus.remove(gpus[0])
        log = open(log_path, 'w')
        log.flush()
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        subproc = subprocess.Popen(shlex.split(command), stdout=log, stderr=log)
        print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} - {command} PID : {subproc.pid}')
        # updates_dict
        pid_to_process_and_gpus[subproc.pid] = (subproc, gpu)
        process_per_gpu[gpu] += 1
    # update free_gpus
    for pid, (current_process, gpu) in pid_to_process_and_gpus.copy().items():
        if poll_process(current_process) is not None:
            print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} - done with {pid}')
            del pid_to_process_and_gpus[pid]
            process_per_gpu[gpu] -= 1
print(f'{datetime.now():%Y/%m/%d-%H:%M:%S}, DONE')