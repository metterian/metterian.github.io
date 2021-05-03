#%%
import sys
def math_block_parser(file_path) -> None:
    with open(file_path, mode="r+", encoding='utf-8') as f:
        lines = f.readlines()

    count = 1
    for i, line in enumerate(lines):
        if line == '$$\n':
            if count:
                lines[i] = '\n$$\n'
                count -= 1
            else:
                lines[i] = '$$\n\n'
                count += 1

    with open(file_path, mode="w+", encoding='utf-8') as f:
        f.writelines(lines)
# %%
if __name__ == '__main__':
    math_block_parser(file_path=sys.argv[1])
