"""
Script1: realAD_gt2ply.py
This script is used to convert the ground truth .txt files to .ply files that can be unsed in blender for conversion to view and .tiffs
"""

import os
# Define the input and output directories
input_dir = r'D:\Project\CPMF\Real3D-AD-PCD\Real3D-AD-PCD\shell\gt'
output_dir = r'output_folder_groundtruth_shell'
marked_dir = os.path.join(output_dir, 'marked')
unmarked_dir = os.path.join(output_dir, 'unmarked')




# Create the output directories if they don't exist
os.makedirs(marked_dir, exist_ok=True)
os.makedirs(unmarked_dir, exist_ok=True)

# Function to process each file
def process_file(file_path, marked_dir, unmarked_dir):
    with open(file_path, 'r') as txt_file:
        lines = txt_file.readlines()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    marked_output_file = os.path.join(marked_dir, base_name + '_marked.ply')
    unmarked_output_file = os.path.join(unmarked_dir, base_name + '_unmarked.ply')

    count_un = 0
    count = 0
    # First pass to count vertices
    for line in lines:
        _, _, _, anomaly = line.strip().split()
        if float(anomaly) != 0.0:
            count += 1
        count_un += 1


    # Create the .ply headers with accurate vertex counts
    header_marked = f"""ply
format ascii 1.0
comment Created by Abhinav
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    header_unmarked = f"""ply
format ascii 1.0
comment Created by Abhinav
element vertex {count_un}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    # Write the headers to the .ply files
    with open(marked_output_file, 'w') as marked_file, open(unmarked_output_file, 'w') as unmarked_file:
        marked_file.write(header_marked)
        unmarked_file.write(header_unmarked)

# Second pass to write vertex data
    # Second pass to write vertex data
# Second pass to write vertex data
    with open(marked_output_file, 'a') as marked_file, open(unmarked_output_file, 'a') as unmarked_file:
        for line in lines:
            x, y, z, anomaly = line.strip().split()
         
            if float(anomaly) == 1.0:
                # If anomaly is 1.0, write the point as grey in the unmarked file
                unmarked_file.write(f"{x} {y} {z} 200 200 200\n")
                # And also write it as white in the marked file
                marked_file.write(f"{x} {y} {z} 255 255 255\n")
            else:
                # If anomaly is 0.0, write the point as white in both files
                unmarked_file.write(f"{x} {y} {z} 255 255 255\n")
                


counter = 0
# Process all .ply files in the directory
for file_name in os.listdir(input_dir):
    try:
        if file_name.endswith('.txt') and any(sub in file_name for sub in ['_sink_cut', '_bulge_cut']):
            counter += 1
            process_file(os.path.join(input_dir, file_name), marked_dir, unmarked_dir)
            print(f"Processed {file_name}", counter)
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")


print("All .ply files have been processed.")

                
