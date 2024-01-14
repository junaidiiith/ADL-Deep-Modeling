import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type=str)
    args = parser.parse_args()

    if args.run_config is None:
        print("Provide a valid run config")
        exit(0)

    lines = [l for l in open(args.run_config).readlines()]
    configs = list()
    for i in range(len(lines)):
        line = lines[i]
        if not len(line.strip()):
            continue
        if line.startswith('python'):
            configs.append((lines[i-1], lines[i]))

    for run_config, run_command in configs:
        print("Running: ", run_config)
        subprocess.run(run_command.split())