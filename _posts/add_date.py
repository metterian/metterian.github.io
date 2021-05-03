#%%

import os, sys, time
import datetime
# %%
""" Append Datetime front of file name """
def append_date(md_files) -> None:
    for md_file in md_files:
        try:
            file_dt = datetime.datetime.strptime(md_file[:10], "%Y-%m-%d")
        except:
            file_dt = ''

        if not file_dt:
            file_path = os.path.join(BASE_DIR, md_file)
            file_dt = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            file_dt = datetime.datetime.strftime(file_dt, "%Y-%m-%d")
            os.rename(file_path, "-".join([file_dt,md_file]))
            print(md_file, "\t-->\t", "-".join([file_dt,md_file]))

# %%
""" Append YAML setting in file """
def get_yaml(title, tags=[], layout = 'post') -> str:
    yaml = "---\n" + "layout: " + layout + "\n" + 'title: ' + '"'+ title+'"' +"\n" + """author: "metterian"\n""" + "tags: " + " ".join(tags) + "\n" + "---\n"
    return yaml

#%%

""" Append YAML setting in file """
def append_yaml(md_files, tags) -> None:
    for md_file in md_files:
        with open(md_file, 'r') as original: data = original.read()
        if "---" in data[:5]:
            continue
        # there is no yaml setting
        else:
            yaml = get_yaml(title=md_file[11:-3], tags=tags)
            with open(md_file, 'w') as modified: modified.write(yaml + data)
            print("YAML Applied: ", md_file)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname((os.path.abspath(__file__)))

    """ Loads Md files """
    md_files = [f for f in os.listdir(BASE_DIR) if os.path.isfile(os.path.join(BASE_DIR, f)) and '.md' in f]
    append_date(md_files)
    md_files = [f for f in os.listdir(BASE_DIR) if os.path.isfile(os.path.join(BASE_DIR, f)) and '.md' in f]
    append_yaml(md_files, tags=sys.argv[1:] )



# %%
