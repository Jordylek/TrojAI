import os
import time
import subprocess
import shlex
from utils import METADATA
from datetime import datetime


# gpu_list = [1,2,3,4,5,6,7]
gpus_per_command = 1
polling_delay_seconds = 10 # Let the time for the model to be loaded
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
free_memory_required = 16000 # in MB # One run on GPT-2 uses 15000MB # One Distilbert run uses 4900MB

gpus_to_leave = {0, 1, 2} # Let this ones free 

path_to_folder = '/scratch/jordan.lekeufack/fine-pruned-weights-linear'
# gpt_linear_models = list(METADATA.loc[METADATA['embedding'] == 'GPT-2'].index)
# gpt_linear_models = set(METADATA.loc[(METADATA['embedding'] == 'GPT-2') & (METADATA['model_architecture'] == 'FCLinear')].index)
linear_models = list(METADATA.loc[METADATA['model_architecture'] == 'FCLinear'].index)
# all_models = set(METADATA.index)
# already_fitted = {s.rstrip('.pt') for s in os.listdir(path_to_folder)}

# models = all_models.difference(already_fitted) # .union(gpt_linear_models)
models = linear_models

commands_to_run = [ ('python3 -u -Wignore pruning.py ' + #  ignore warnings
                   f'--model_id {model_id} '+
                   f'--path_to_folder {path_to_folder} ' +
                   f'--prune_low_weights True ' # To remove if not pruning low weights for FCLinear
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
print(f'Starting... {start_time:%Y/%m/%d-%H:%M:%S}, Commands to run : {len(commands_to_run)}')
while len(commands_to_run) > 0 or len(pid_to_process_and_gpus) > 0:
    time.sleep(polling_delay_seconds)
    gpus_memory = get_gpu_memory_map()
    free_gpus = {gpu for gpu in gpus_memory if (gpus_memory[gpu] >= free_memory_required) and not(gpu in gpus_to_leave)}
    now = datetime.now()
    if (now - prev_message_time).seconds >  delay_between_messages_seconds:
        print(f'{now:%Y/%m/%d-%H:%M:%S}, left to run : {len(commands_to_run)}, running : {len(pid_to_process_and_gpus)}')
        prev_message_time = now

    while len(free_gpus) >= gpus_per_command:
        # print(f'free_gpus: {free_gpus}')
        # gpus_memory = get_gpu_memory_map()
        gpu = free_gpus.pop()
        # gpus = [gpu for gpu in gpus_memory if (gpus_memory[gpu] >= free_memory_required) and (gpu in free_gpus)]
        # gpus = []
        # for i in range(gpus_per_command):
        #     # updates free_gpus
        #     gpus.append(str(free_gpus.pop()))
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
    # update free_gpus
    for pid, (current_process, gpu) in pid_to_process_and_gpus.copy().items():
        if poll_process(current_process) is not None:
            print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} - done with {pid}')
            # free_gpus.add(gpu)# update(gpus)
            del pid_to_process_and_gpus[pid]
            
print(f'{datetime.now():%Y/%m/%d-%H:%M:%S}, DONE')