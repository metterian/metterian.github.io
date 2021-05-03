#%%

import os, sys, time
import datetime
BASE_DIR = os.path.dirname((os.path.abspath(__file__)))


# %%
from os import listdir
from os.path import isfile, join
md_files = [f for f in listdir(BASE_DIR) if isfile(join(BASE_DIR, f))]

# %%
for md_file in md_files:
    try:
        file_dt = datetime.datetime.strptime(md_file[:10], "%Y-%m-%d")
    except:
        file_dt = ''

    if not file_dt and '.md' in md_file:
        file_path = os.path.join(BASE_DIR, md_file)
        file_dt = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        file_dt = datetime.datetime.strftime(file_dt, "%Y-%m-%d")
        os.rename(file_path, "-".join([file_dt,md_file]))

# %%
