import subprocess

def run(cmd):
    print(f"\n Running: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    run("python ./stages/sam_full_image.py")
    run("python ./stages/mask_cleaning.py")
    run("python ./stages/build_label_map.py")
    run("python ./stages/split_dataset.py")
    run("python ./stages/train_unet.py")
    run("python ./stages/evaluate_mious.py")

    print("\n[SUCCESS] Pipeline finished successfully!")

if __name__ == "__main__":
    main()
