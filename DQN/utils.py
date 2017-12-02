
from operator import add
def move(center,action,i):
    if action == 0:
        center[0] += i
    if action == 1:
        center[0] -= i
    if action == 2:
        center[1] += i
    if action == 3:
        center[1] -= i
    new_center = center
    if new_center[1] > 63:
        new_center[1] -= i
    if new_center[0] > 63:
        new_center[0] -= i
    if new_center[1] < 0:
        new_center[1] += i
    if new_center[0] < 0:
        new_center[0] += i
    return new_center


