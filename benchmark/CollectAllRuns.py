import os
import shutil

src_dir = "./recipes"
dst_dir = "./archived_runs"

for d in os.listdir(src_dir):

    if "_" in d or "." in d:
        continue

    print(d)

    c_src = f"{src_dir}/{d}/runs/"
    print(c_src)

    c_dst = f"{dst_dir}/{d}/runs/"
    print(c_dst)
    os.makedirs(c_dst, exist_ok=True)

    for _file in [f for f in os.listdir(c_src) if ".json" in f]:
        shutil.copyfile(f"{src_dir}/{d}/runs/{_file}", f"{dst_dir}/{d}/runs/{_file}")
