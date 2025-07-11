import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import tkinter as tk
from tkinter import filedialog

def choose_nifti_file():
    import tkinter as tk
    from tkinter import filedialog

    # Step 1: Prompt for folder
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with NIfTI Files")
    if not folder_path:
        print("No folder selected.")
        return None

    # Step 2: Get NIfTI files
    nifti_files = [f for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
    if not nifti_files:
        print("No .nii or .nii.gz files found.")
        return None

    # Step 3: Use a new window to choose one safely
    def select_file():
        selected = var.get()
        if selected:
            nonlocal selected_file
            selected_file = os.path.join(folder_path, selected)
            root.quit()

    root = tk.Tk()
    root.title("Select a NIfTI File")
    var = tk.StringVar(root)
    var.set(nifti_files[0])
    dropdown = tk.OptionMenu(root, var, *nifti_files)
    dropdown.pack(padx=20, pady=10)
    tk.Button(root, text="Open", command=select_file).pack(pady=10)
    selected_file = None
    root.mainloop()
    return selected_file

def nifti_slider_viewer(file_path, plane='axial'):
    img = nib.load(file_path)
    data = img.get_fdata()

    if plane == 'axial':
        axis = 2
    elif plane == 'coronal':
        axis = 1
    elif plane == 'sagittal':
        axis = 0
    else:
        raise ValueError("Plane must be 'axial', 'coronal', or 'sagittal'")

    num_slices = data.shape[axis]
    init_slice = num_slices // 2

    def get_slice(index):
        return np.take(data, index, axis=axis)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    slice_2d = get_slice(init_slice)
    img_plot = ax.imshow(np.rot90(slice_2d), cmap='gray')
    ax.set_title(f"{plane.capitalize()} Slice #{init_slice}")
    ax.axis('off')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=init_slice, valfmt='%d')

    def update(val):
        idx = int(slider.val)
        new_slice = get_slice(idx)
        img_plot.set_data(np.rot90(new_slice))
        ax.set_title(f"{plane.capitalize()} Slice #{idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    file = choose_nifti_file()
    if file:
        nifti_slider_viewer(file, plane='axial')  # you can change to 'sagittal' or 'coronal'
