import sys, os, json, time
from random import randint
import argparse

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.channels import SSHChannel
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor

from parsl.addresses import address_by_hostname
#from parsl.monitoring.monitoring import MonitoringHub

import parsl_utils

with open('executors.json', 'r') as f:
    exec_conf = json.load(f)


for label,executor in exec_conf.items():
    for k,v in executor.items():
        if type(v) == str:
            exec_conf[label][k] = os.path.expanduser(v)


# Job runs in directory /pw/jobs/job-number
job_number = os.path.dirname(os.getcwd().replace('/pw/jobs/', ''))

# PARSL APPS:
@parsl_utils.parsl_wrappers.log_app
@python_app(executors=['myexecutor_1'])
def hello_python_app_1(name = '', stdout='std.out', stderr = 'std.err'):
    import socket
    if not name:
        name = 'python_app_1'
    return 'Hello ' + name + ' from ' + socket.gethostname()

@parsl_utils.parsl_wrappers.log_app
@parsl_utils.parsl_wrappers.stage_app(exec_conf['myexecutor_1']['HOST_USER'] + '@' + exec_conf['myexecutor_1']['HOST_IP'])
@bash_app(executors=['myexecutor_1'])
def hello_srun_1(run_dir, slurm_info = {}, inputs_dict = {}, outputs_dict = {}, stdout='std.out', stderr = 'std.err'):
    if not slurm_info:
        slurm_info = {
            'nodes': '1',
            'partition': 'compute',
            'ntasks_per_node': '1',
            'walltime': '01:00:00'
        }

    return '''
        cd {run_dir}
        cat {hello_in} > {hello_out}
        srun --nodes={nodes}-{nodes} --partition={partition} --ntasks-per-node={ntasks_per_node} --time={walltime} --exclusive hostname >> {hello_out}
    '''.format(
        run_dir = run_dir,
        hello_in = inputs_dict["test-in-file"]["worker_path"],
        hello_out = outputs_dict["test-out-file"]["worker_path"],
        nodes = slurm_info['nodes'],
        partition = slurm_info['partition'],
        ntasks_per_node = slurm_info['ntasks_per_node'],
        walltime = slurm_info['walltime']
    )

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    pwargs=vars(parser.parse_args())
    print(pwargs)
    return pwargs

if __name__ == '__main__':
    args = read_args()
    job_number = args['job_number']

    # Add sandbox directory
    for exec_label, exec_conf_i in exec_conf.items():
        if 'RUN_DIR' in exec_conf_i:
            exec_conf[exec_label]['RUN_DIR'] = os.path.join(exec_conf_i['RUN_DIR'])
        else:
            base_dir = '/tmp'
            exec_conf[exec_label]['RUN_DIR'] = os.path.join(base_dir, str(job_number))

    config = Config(
        executors = [
            HighThroughputExecutor(
                worker_ports = ((int(exec_conf['myexecutor_1']['WORKER_PORT_1']), int(exec_conf['myexecutor_1']['WORKER_PORT_2']))),
                label = 'myexecutor_1',
                worker_debug = True,             # Default False for shorter logs
                cores_per_worker = float(exec_conf['myexecutor_1']['CORES_PER_WORKER']), # One worker per node
                worker_logdir_root = exec_conf['myexecutor_1']['WORKER_LOGDIR_ROOT'],  #os.getcwd() + '/parsllogs',
                provider = LocalProvider(
                    worker_init = 'source {conda_sh}; conda activate {conda_env}; cd {run_dir}'.format(
                        conda_sh = os.path.join(exec_conf['myexecutor_1']['CONDA_DIR'], 'etc/profile.d/conda.sh'),
                        conda_env = exec_conf['myexecutor_1']['CONDA_ENV'],
                        run_dir = exec_conf['myexecutor_1']['RUN_DIR']
                    ),
                    channel = SSHChannel(
                        hostname = exec_conf['myexecutor_1']['HOST_IP'],
                        username = exec_conf['myexecutor_1']['HOST_USER'],
                        script_dir = exec_conf['myexecutor_1']['SSH_CHANNEL_SCRIPT_DIR'], # Full path to a script dir where generated scripts could be sent to
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

    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    print('\n\n\nHELLO FROM CONTROLLER NODE:', flush = True)
    fut_1 = hello_python_app_1(name = args['name'])

    print(fut_1.result())

    print('\n\n\nHELLO FROM COMPUTE NODES:', flush = True)
    print('\n\nmyexecutor_1:', flush = True)
    fut_1 = hello_srun_1(
        run_dir = exec_conf['myexecutor_1']['RUN_DIR'],
        slurm_info = {
            'nodes': exec_conf['myexecutor_1']['NODES'],
            'partition': exec_conf['myexecutor_1']['PARTITION'],
            'ntasks_per_node': exec_conf['myexecutor_1']['NTASKS_PER_NODE'],
            'walltime': exec_conf['myexecutor_1']['WALLTIME']
        },
        inputs_dict = {
            "test-in-file": {
                "type": "file",
                "global_path": "pw://{cwd}/hello_srun.in",
                "worker_path": "{remote_dir}/hello_srun.in".format(remote_dir =  exec_conf['myexecutor_1']['RUN_DIR'])
            }
        },
        outputs_dict = {
            "test-out-file": {
                "type": "file",
                "global_path": "pw://{cwd}/hello_srun-1.out",
                "worker_path": "{remote_dir}/hello_srun-1.out".format(remote_dir =  exec_conf['myexecutor_1']['RUN_DIR'])
            }
        },
        stdout = os.path.join(exec_conf['myexecutor_1']['RUN_DIR'], 'std.out'),
        stderr = os.path.join(exec_conf['myexecutor_1']['RUN_DIR'], 'std.err')
    )

    print(fut_1.result())
