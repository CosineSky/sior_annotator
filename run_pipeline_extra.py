import subprocess

def run(cmd):
    print(f"\n Running: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    run("python ./stages-extra/parse_gray_label.py")
    run("python ./stages-extra/mask_cleaning.py")
    run("python ./stages-extra/split_dataset.py")
    run("python ./stages-extra/train_model.py")
    run("python ./stages/evaluate_mious.py")

    print("\n[SUCCESS] Pipeline finished successfully!")

if __name__ == "__main__":
    main()
