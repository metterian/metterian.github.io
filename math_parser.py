#%%
with open("_posts/2021-05-01-🧐대체 시그모이드(Sigmoid) 함수가 뭔데.md", mode="r+", encoding='utf-8') as f:
    lines = f.readlines()

# %%
count = 1
for i, line in enumerate(lines):
    if line == '$$\n':
        if count:
            lines[i] = '\n$$\n'
            count -= 1
        else:
            lines[i] = '$$\n\n'
            count += 1



# %%
with open("_posts/2021-05-01-🧐대체 시그모이드(Sigmoid) 함수가 뭔데.md", mode="w+", encoding='utf-8') as f:
    f.writelines(lines)
# %%
