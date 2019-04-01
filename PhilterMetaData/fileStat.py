from __future__ import print_function

import argparse
from datetime import datetime as dt
import os
import sys

parser = argparse.ArgumentParser(description='File metadata info')

parser.add_argument("FILE_PATH", help="Path to file")

args = parser.parse_args()
file_path = args.FILE_PATH
stat_info = os.stat(file_path)
major = os.major(stat_info.st_dev)
minor = os.minor(stat_info.st_dev)

if "linux" in sys.platform or "darwin" in sys.platform:
    print("Change time", dt.fromtimestamp(stat_info.st_ctime))
elif "win" in sys.platform:
    print("Creation time: ", dt.fromtimestamp(stat_info.st_ctime))
else:
    print("[-]Unsupported platform {} detected. Cannot interpret ""creation/ change timestamp.".format(sys.platform))

print("Modification time:", dt.fromtimestamp(stat_info.st_mtime))
print("Access time: ", dt.fromtimestamp(stat_info.st_atime))
print("File mode:", stat_info.st_mode)
print("File inode:", stat_info.st_ino)
print("Device ID:", stat_info.st_dev)
print("\tMajor: ", major)
print("\tMinor:", minor)
print("Is a symlink: ", os.path.islink(file_path))
print("Absolute Path: ", os.path.abspath(file_path))
print("File exists: ", os.path.exists(file_path))
print("Parent directory:", os.path.dirname(file_path))
print("Parent directory: {} | File name : {}".format(*os.path.split(file_path)))


