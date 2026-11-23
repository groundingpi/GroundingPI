"""Run a YAML entrypoint from the project root; --dry-run never imports model code."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
from config_contract import (relative as checked_relative, validate_native,
                             validate_tree, checkpoint_args, native_inputs, validate_argv)
TARGETS={'vlm-vllm': ['-m', 'infer.serve_vllm'], 'vlm-train': ['-m', 'swift.cli.sft'], 'vlm-config': ['-m', 'train.vlm.config_runtime'], 'vlm-serve': ['-m', 'infer.serve_vlm'], 'vlm-serve-xpu': ['-m', 'infer.serve_vlm_xpu'], 'vlm-overlay': ['infer/overlay/prepare_model_overlay.py'], 'eval': ['-m', 'eval.eval_runner'], 'eval-yaml': ['scripts/evaluate.py']}

def relative(value):
    return checked_relative(ROOT,value)

@contextmanager
def project_context():
    previous=Path.cwd()
    sys.path.insert(0,str(ROOT))
    try:
        os.chdir(ROOT)
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(ROOT))

def build_command(cfg, *, validate_training=False):
    entry=cfg["entrypoint"]
    target=TARGETS[entry]
    checkpoint_argv=checkpoint_args(ROOT,cfg)
    validate_argv(ROOT,cfg.get("args",[]))
    argv=list(cfg.get("args",[]))
    training=entry == "vlm-train"
    if not training:
        if "native_config" in cfg or "distributed" in cfg:
            raise ValueError("native_config/distributed are supported only for training")
        return [sys.executable,*target,*argv]
    if argv:
        raise ValueError("training parameters must come from native_config, not args overrides")
    if not isinstance(cfg.get("native_config"),str):
        raise ValueError("training requires a project-relative native_config YAML")
    recipe=relative(cfg["native_config"])
    native=yaml.safe_load(recipe.read_text())
    if not isinstance(native,dict):raise ValueError("native_config must contain a mapping")
    topology=cfg.get("distributed",{})
    if not isinstance(topology,dict) or set(topology)-{"nproc_per_node","nnodes","node_rank","master_addr","master_port"}:
        raise ValueError("invalid distributed configuration")
    nproc=topology.get("nproc_per_node",1);nodes=topology.get("nnodes",1);rank=topology.get("node_rank",0)
    if any(type(x) is not int for x in (nproc,nodes,rank)) or nproc<1 or nodes<1 or not 0<=rank<nodes:
        raise ValueError("invalid distributed topology")
    declared=native.get("runtime",{}).get("expected_world_size")
    if declared is not None and int(declared)!=nproc*nodes:
        raise ValueError("native_config expected_world_size does not match distributed topology")
    validate_native(ROOT,native,nproc*nodes)
    with project_context():
        from train.release_parameters import validate_training_fields
        validate_training_fields(native)
    with project_context():
        from train.vlm import config_runtime
        from train.vlm.portable_validation import validate
        validate(recipe,expected_world_size=nproc*nodes,check_resources=validate_training)
        argv=config_runtime.as_cli(recipe,None)
        if checkpoint_argv:
            if native["training"].get("resume_from_checkpoint"):
                raise ValueError("VLM resume must be specified in either checkpoint or native training, not both")
            argv.extend(checkpoint_argv)
    command=[sys.executable,"-m","torch.distributed.run",f"--nproc_per_node={nproc}"]
    if nodes==1:
        command.append("--standalone")
    else:
        address=topology.get("master_addr");port=topology.get("master_port",29500)
        if not isinstance(address,str) or not address or type(port) is not int or not 1<=port<=65535:
            raise ValueError("multi-node training requires master_addr and a valid master_port")
        command.extend([f"--nnodes={nodes}",f"--node_rank={rank}",f"--master_addr={address}",f"--master_port={port}"])
    return [*command,*target,*argv]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    cfg=yaml.safe_load(relative(args.config).read_text())
    if not isinstance(cfg,dict):raise ValueError("config must be a mapping")
    unknown=set(cfg)-{"entrypoint","args","env","inputs","outputs","native_config","distributed","checkpoint"}
    if unknown:raise ValueError(f"unknown keys: {sorted(unknown)}")
    validate_tree(ROOT,cfg)
    target=TARGETS[cfg["entrypoint"]]
    argv=cfg.get("args",[])
    env=cfg.get("env",{})
    if not isinstance(argv,list) or not all(isinstance(x,str) for x in argv):raise ValueError("args must be a string list")
    if not isinstance(env,dict):raise ValueError("env must be a mapping")
    for value in [*argv,*map(str,env.values())]:
        if value.startswith("/") or "=/" in value or ".." in Path(value.split("=", 1)[-1]).parts:raise ValueError("absolute configuration paths are not allowed")
    for key in ("inputs","outputs"):
        if not isinstance(cfg.get(key,[]),list) or not all(isinstance(v,str) for v in cfg.get(key,[])):
            raise ValueError(f"{key} must be a list of relative paths")
    for value in cfg.get("inputs",[])+cfg.get("outputs",[]):relative(value)
    if cfg.get("native_config"):
        native_runtime=yaml.safe_load(relative(cfg["native_config"]).read_text()).get("runtime",{})
        image_limit=native_runtime.get("image_max_token_num")
        if image_limit is not None:
            if "IMAGE_MAX_TOKEN_NUM" in env and str(env["IMAGE_MAX_TOKEN_NUM"])!=str(image_limit):
                raise ValueError("IMAGE_MAX_TOKEN_NUM differs from native runtime.image_max_token_num")
            env=dict(env,IMAGE_MAX_TOKEN_NUM=str(image_limit))
    cmd=build_command(cfg)
    required=list(cfg.get("inputs",[]))+list(cfg.get("checkpoint",{}).values())
    if cfg.get("native_config"):
        required+=native_inputs(yaml.safe_load(relative(cfg["native_config"]).read_text()))
    required=list(dict.fromkeys(required))
    missing=[x for x in required if not relative(x).exists()]
    print(json.dumps({"command":cmd,"env":env,"missing_inputs":missing},indent=2))
    if args.dry_run:return
    if missing:raise SystemExit("Provide missing inputs before execution.")
    child_env=os.environ.copy()
    child_env.update({k:str(v) for k,v in env.items()})
    child_env["PYTHONPATH"]=str(ROOT)+os.pathsep+child_env.get("PYTHONPATH","")
    previous_env=os.environ.copy()
    try:
        os.environ.update(child_env)
        cmd=build_command(cfg,validate_training=True)
    finally:
        os.environ.clear();os.environ.update(previous_env)
    raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=child_env))

if __name__=="__main__":main()
