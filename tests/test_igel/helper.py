import logging
import os
import shutil


def remove_file(f):
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
