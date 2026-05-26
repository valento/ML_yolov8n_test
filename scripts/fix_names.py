import os

def rename_labels():
    # Path to your labels directory
    labels_dir = "dataset/train/labels"
    
    if not os.path.exists(labels_dir):
        print(f"Error: Could not find directory at {labels_dir}")
        return

    # List all files in the labels folder
    files = os.listdir(labels_dir)
    rename_count = 0

    print("Checking labels for typos...")

    for filename in files:
        # Check if the file starts with your typo string 'stck_'
        if filename.startswith("stck_") and filename.endswith(".txt"):
            # Construct the corrected filename by replacing 'stck_' with 'stick_'
            new_filename = filename.replace("stck_", "stick_")
            
            # Form full absolute paths
            old_path = os.path.join(labels_dir, filename)
            new_path = os.path.join(labels_dir, new_filename)
            
            # Rename the file on your hard drive
            os.rename(old_path, new_path)
            rename_count += 1

    print(f"✨ Successfully renamed {rename_count} label files from 'stck_' to 'stick_'!")

if __name__ == "__main__":
    rename_labels()
