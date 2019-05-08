from enum import Enum
from collections import defaultdict

class Action(Enum):
    REST = 0
    CLIMB = 1
    DOWN = 2

class Position(Enum):
    BOTTOM = 0
    MIDDLE = 1
    TOP = 2


shape = defaultdict();

shape[Position.BOTTOM] = [[Action.REST, 0.1, Position.BOTTOM], [Action.CLIMB, 0, Position.TOP],
                               [Action.DOWN, -1, Position.BOTTOM]]
shape[Position.MIDDLE] = [[Action.REST, -1, Position.MIDDLE], [Action.CLIMB, 0.3, Position.TOP],
                               [Action.DOWN, 1, Position.BOTTOM]]
shape[Position.TOP] = [[Action.REST, 0.1, Position.TOP], [Action.CLIMB, -1, Position.TOP],
                            [Action.DOWN, 0.2, Position.MIDDLE]]

print(shape)

print(shape[Position.BOTTOM])
print(shape[Position(0)])

print(shape[Position.BOTTOM][0])
# reward of Position Bottom - Action Rest
print(shape[Position.BOTTOM][Action.REST.value][1])
# next state of Position Bottom - Action Rest
print(shape[Position.BOTTOM][Action.REST.value][2])
