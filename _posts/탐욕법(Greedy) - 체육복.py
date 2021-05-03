#%%
n = 5
lost = [2, 4]
reserve = [1, 3, 5]

# %%
def solution(n, lost, reserve):
    u = [1] * (n+2) # 처음과 끝애 dummy 배열 추가
    for i in reserve:
        u[i] += 1
    for i in lost:
        u[i] -= 1

    for i in range(1, n+1):
        # 왼쪽 친구가 체육복이 없고 내가 빌려줄 수 있으면
        if u[i-1] == 0 and u[i] == 2:
            u[i-1:i+1] = [1,1]
        # 내가 체육복을 가지고 있고, 오르쪽 사람이 체육복이 없으면 빌려준다
        elif u[i] == 2 and u[i+1] ==0:
            u[i:i+2] = [1,1]

    return len([x for x in u[1:n+1] if x > 0])

solution(n, lost, reserve)
# %%

def solution(n, lost, reserve):
    # 체육복을 가져왔는데 도난 당한 사람을 먼저 파악
    s = set(lost) & set(reserve) # &는 교집합을 뜻함
    l = set(lost) - s # 차집합, 체육복을 빌려야 하는 학생들
    r = set(reserve) - s # 체육복을 빌려줄 수 있는 학생들

    for x in sorted(r):
        # x는 체육복을 빌려 줄 수 있는 학생
        if x - 1 in l: # 왼쪽사람이 체육복이 없으면
            l.remove(x - 1)
        elif x + 1 in l:
            l.remove(x + 1)

        return n - len(l)
solution(n, lost, reserve)

# %%
