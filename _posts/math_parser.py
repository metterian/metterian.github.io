#%%
import sys
def math_block_parser(file_path) -> None:
    with open(file_path, mode="r+", encoding='utf-8') as f:
        lines = f.readlines()

    count = 1
    double = 0
    for i, line in enumerate(lines):
        if line == '$$\n':
            if lines[i-1] == '\n' or lines[i+1] == '\n':
                continue
            if count:
                lines[i] = '\n$$\n'
                count -= 1
            else:
                lines[i] = '$$\n'
                count += 1
        # elif line == '> $$\n' or line == '>$$\n':
        #     if lines[i-1] == '> \n' or lines[i-1] == '>\n' or lines[i-1]=='>':
        #         continue
        #     if not double:
        #         lines[i] = '> \n' + line
        #         double += 1
        #     elif double == 1:
        #         double -= 1
        #         continue
    with open(file_path, mode="w+", encoding='utf-8') as f:
        f.writelines(lines)



if __name__ == '__main__':
    math_block_parser(file_path=sys.argv[1])
