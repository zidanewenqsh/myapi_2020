import os
with open(r"installed.txt") as f:
    for line in f.readlines():
        try:
            print(line.strip())
            os.remove(line.strip())
        except:
            pass