import subprocess


if __name__ == "__main__":
    lines = [l for l in open('run_configs.txt').readlines()]
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