from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.providers import LocalProvider, SlurmProvider
from parsl.executors import HighThroughputExecutor
#from parsl.monitoring.monitoring import MonitoringHub
from parsl.addresses import address_by_hostname
import os,json

# Need to name the job to be able to remove it with clean_resources.sh!
job_number = os.getcwd().replace('/pw/jobs/', '')

with open('executors.json', 'r') as f:
    exec_conf = json.load(f)

for label,executor in exec_conf.items():
    for k,v in executor.items():
        if type(v) == str:
            exec_conf[label][k] = os.path.expanduser(v)

# Add sandbox directory
for exec_label, exec_conf_i in exec_conf.items():
    if 'RUN_DIR' in exec_conf_i:
        exec_conf[exec_label]['RUN_DIR'] = os.path.join(exec_conf_i['RUN_DIR'])
    else:
        base_dir = '/tmp'
        exec_conf[exec_label]['RUN_DIR'] = os.path.join(base_dir, str(job_number))

# PARSL CONFIGURATION
# - controller:        For apps that run in the controller node (prepare run directory)
# - compute_partition: For apps that run in the compute nodes (training)
config = Config(
    executors = [
        HighThroughputExecutor(
            worker_ports = ((int(exec_conf['controller']['WORKER_PORT_1']), int(exec_conf['controller']['WORKER_PORT_2']))),
            label = 'controller',
            worker_debug = True,             # Default False for shorter logs
            cores_per_worker = float(exec_conf['controller']['CORES_PER_WORKER']), # One worker per node
            worker_logdir_root = exec_conf['controller']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
            provider = LocalProvider(
                worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                    conda_sh = os.path.join(exec_conf['controller']['CONDA_DIR'], 'etc/profile.d/conda.sh'),
                    conda_env = exec_conf['controller']['CONDA_ENV'],
                    run_dir = exec_conf['controller']['RUN_DIR']
                ),
                channel = SSHChannel(
                    hostname = exec_conf['controller']['HOST_IP'],
                    username = exec_conf['controller']['HOST_USER'],
                    script_dir = exec_conf['controller']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                    key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                )
            )
        ),
        HighThroughputExecutor(
            worker_ports = ((int(exec_conf['compute_partition']['WORKER_PORT_1']), int(exec_conf['compute_partition']['WORKER_PORT_2']))),
            label = 'compute_partition',
            worker_debug = True,             # Default False for shorter logs
            cores_per_worker = float(exec_conf['compute_partition']['CORES_PER_WORKER']), # One worker per node
            worker_logdir_root = exec_conf['compute_partition']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
            address = exec_conf['compute_partition']['ADDRESS'],
            provider = SlurmProvider(
                partition = exec_conf['compute_partition']['PARTITION'],
                nodes_per_block = int(exec_conf['compute_partition']['NODES_PER_BLOCK']),
                cores_per_node = int(exec_conf['compute_partition']['NTASKS_PER_NODE']),
                min_blocks = int(exec_conf['compute_partition']['MIN_BLOCKS']),
                max_blocks = int(exec_conf['compute_partition']['MAX_BLOCKS']),
                walltime = exec_conf['compute_partition']['WALLTIME'],
                worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                    conda_sh = os.path.join(exec_conf['compute_partition']['CONDA_DIR'], 'etc/profile.d/conda.sh'),
                    conda_env = exec_conf['compute_partition']['CONDA_ENV'],
                    run_dir = exec_conf['compute_partition']['RUN_DIR']
                ),
                channel = SSHChannel(
                    hostname = exec_conf['compute_partition']['HOST_IP'],
                    username = exec_conf['compute_partition']['HOST_USER'],
                    script_dir = exec_conf['compute_partition']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
                    key_filename = '/home/{PW_USER}/.ssh/pw_id_rsa'.format(PW_USER = os.environ['PW_USER'])
                )
            )
        )
    ]
)

#,
#    monitoring = MonitoringHub(
#       hub_address = address_by_hostname(),
#       resource_monitoring_interval = 5
#   )
#)
