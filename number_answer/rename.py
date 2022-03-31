import os
import sys

source = sys.path[0] + "/media/1/train/"

def insert_x(command):
    for i, c in enumerate(command):
        if c.isdigit():
            break
    x = command[:i] + '_' + command[i:]
    os.rename(source + command,source + x)


if __name__ == "__main__":
    total_files = os.listdir(source)
    total_files_m = []

    for x in total_files:
        if x.endswith(".mp4"):
            #name = re.sub(r"_.*\.mp4", "", x)
            total_files_m.append(x)

    for x in total_files_m:
        insert_x(x)
