#!/usr/bin/env python3
"""
checker.py

Audit SLURM cluster resources, OS limits, and environment—all locally.

Usage: 
    ./checker.py [--output FILE]

If --output is given, writes JSON to that file; otherwise prints to stdout.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime

def run(cmd):
    """Run cmd (list) and return stdout or ''."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def gather_slurm_overview():
    # Try JSON sinfo (SLURM ≥20.02); else plain text fallback
    out = run(["sinfo", "--json"])
    if out:
        try:
            data = json.loads(out).get("data", [])
            return data
        except json.JSONDecodeError:
            pass
    # fallback
    lines = run(["sinfo", "-N", "-o", "%N %c %G %P"]).splitlines()[1:]
    entries = []
    for l in lines:
        node, cpus, gres, part = l.split(maxsplit=3)
        entries.append({
            "node": node,
            "cpus_state": cpus,
            "gres": gres,
            "partition": part
        })
    return entries

def gather_slurm_configs(nodes, partitions):
    cfg = {}
    cfg["global"] = run(["scontrol", "show", "config"])
    cfg["partitions"] = {p: run(["scontrol", "show", "partition", p]) for p in partitions}
    cfg["nodes"] = {n: run(["scontrol", "show", "node", n]) for n in nodes}
    return cfg

def gather_os_info():
    info = {}
    info["timestamp"] = datetime.utcnow().isoformat() + "Z"
    info["uname"] = run(["uname", "-a"])
    info["lscpu"] = run(["lscpu"])
    info["memory_free"] = run(["free", "-h"])
    info["df"] = run(["df", "-h"])
    return info

def gather_limits():
    limits = {}
    limits["ulimit"] = run(["bash", "-lc", "ulimit -a"])
    limits["limits_conf"] = run(["cat", "/etc/security/limits.conf"])
    return limits

def gather_modules():
    # Works if you have module (Environment Modules or Lmod)
    modules = {}
    modules["available"] = run(["bash", "-lc", "module avail 2>&1"])
    modules["loaded"]    = run(["bash", "-lc", "module list 2>&1"])
    return modules

def main():
    p = argparse.ArgumentParser(description="Cluster checker (local)")
    p.add_argument("--output", "-o", help="Write JSON to file")
    args = p.parse_args()

    # 1) SLURM overview
    sinfo = gather_slurm_overview()
    nodes = {e.get("node") or e.get("nodehost") for e in sinfo}
    partitions = {e.get("partition") for e in sinfo}

    # 2) SLURM configs
    slurm_cfg = gather_slurm_configs(nodes, partitions)

    # 3) OS + FS info
    os_info = gather_os_info()

    # 4) Limits
    limits = gather_limits()

    # 5) Modules
    modules = gather_modules()

    report = {
        "slurm_overview": sinfo,
        "slurm_configs": slurm_cfg,
        "os_info": os_info,
        "limits": limits,
        "modules": modules
    }

    if args.output:
        with open(args.output, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"Written report to {args.output}")
    else:
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
