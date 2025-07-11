import os

def delete_label_files(folder):
    deleted = 0
    for filename in os.listdir(folder):
        if filename.endswith('.label.nii.gz'):
            file_path = os.path.join(folder, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            deleted += 1
    print(f"\nDone. Deleted {deleted} label files.")

# 🔁 Replace with your actual folder path
your_folder = r"C:\Users\HenryLi\Downloads\200 good scans"
delete_label_files(your_folder)
