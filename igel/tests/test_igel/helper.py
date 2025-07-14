import logging
import os
import shutil


def remove_file(f: str) -> None:
    """
    Remove a file at the given path if it exists.

    Args:
        f (str): The path to the file to be removed.
    """
    try:
        if os.path.exists(f):
            os.remove(f)
    except Exception as ex:
        print(ex)


def remove_folder(folder, all_content=True):
    try:
        if all_content:
            shutil.rmtree(folder)
        else:
            os.rmdir(folder)
    except Exception as ex:
        print(ex)
