
from operator import add
def move(center,action):
    if action == 0:
        center[0] += 4
    if action == 1:
        center[0] -= 4
    if action == 2:
        center[1] += 4
    if action == 3:
        center[1] -= 4
    new_center = center
    if new_center[1] > 63:
        new_center[1] = 63
    if new_center[0] > 63:
        new_center[0] = 63
    if new_center[1] < 0:
        new_center[1] = 0
    if new_center[0] < 0:
        new_center[0] = 0
    return new_center


