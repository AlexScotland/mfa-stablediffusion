import os

def get_root_folder():
    # Get the absolute path of the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Split the path into the directory and the last component
    parent_dir, _ = os.path.split(script_dir)

    return parent_dir