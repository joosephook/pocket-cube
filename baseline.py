import numpy as np
from collections import defaultdict
from time import perf_counter_ns
from itertools import islice

"""
Here we define the accessor functions of the cube.
Each function returns a view to some data backed by
a 2D numpy array which we will to represent the state
of our cube.
"""


def front(cube):
    return cube.reshape(2, 12)[:, 0:2]


def right(cube):
    return cube.reshape(2, 12)[:, 2:4]


def down(cube):
    return cube.reshape(2, 12)[:, 4:6]


def left(cube):
    return cube.reshape(2, 12)[:, 6:8]


def top(cube):
    return cube.reshape(2, 12)[:, 8:10]


def back(cube):
    return cube.reshape(2, 12)[:, 10:12]


def fresh_cube():
    """
    Return a fresh, solved 2x2 cube
    """
    cube = np.arange(24, dtype=np.uint8).reshape(2, 12) + 1
    front(cube)[:] = [[4, 1], [3, 2]]

    right(cube)[:] = [[8, 5], [7, 6]]

    down(cube)[:] = [[12, 9], [11, 10]]

    left(cube)[:] = [[16, 13], [15, 14]]

    top(cube)[:] = [[20, 17], [19, 18]]

    back(cube)[:] = [[24, 21], [23, 22]]

    return cube.ravel()

EMPTY = {0: '  '}
BLACK = {num + 1: "\U00002B1B" for num in range(0, 4)}
RED = {num + 1: "\U0001F7E5" for num in range(4, 8)}
GREEN = {num + 1: "\U0001F7E9" for num in range(8, 12)}
ORANGE = {num + 1: "\U0001F7E7" for num in range(12, 16)}
BLUE = {num + 1: "\U0001F7E6" for num in range(16, 20)}
YELLOW = {num + 1: "\U0001F7E8" for num in range(20, 24)}

UNICODE = EMPTY | BLACK | RED | GREEN | ORANGE | BLUE | YELLOW


def display(cube):
    cubetangle = np.zeros((6, 8), dtype=np.uint8)

    cubetangle[2:4, 2:4] = front(cube)
    cubetangle[2:4, 4:6] = right(cube)
    cubetangle[4:6, 2:4] = down(cube)
    cubetangle[2:4, 0:2] = left(cube)
    cubetangle[0:2, 2:4] = top(cube)
    cubetangle[2:4, 6:8] = back(cube)

    out = np.empty((6, 8), dtype=object)
    lines = []
    for i in range(6):
        for j in range(8):
            out[i, j] = UNICODE[cubetangle[i, j]]
        lines.append(''.join(out[i]))
    return '\n'.join(lines)


# In[5]:


def front(cube):
    return cube.reshape(2, 12)[:, 0:2]


def right(cube):
    return cube.reshape(2, 12)[:, 2:4]


def down(cube):
    return cube.reshape(2, 12)[:, 4:6]


def left(cube):
    return cube.reshape(2, 12)[:, 6:8]


def top(cube):
    return cube.reshape(2, 12)[:, 8:10]


def back(cube):
    return cube.reshape(2, 12)[:, 10:12]

def B(cube):
    new_cube = cube.copy()
    right(new_cube)[:, 1] = down(cube)[1][::-1]
    down(new_cube)[1:2, :] = left(cube)[:, 0]
    left(new_cube)[:, 0] = top(cube)[0][::-1]
    top(new_cube)[0:1, :] = right(cube)[:, 1]
    back(new_cube)[:] = np.rot90(back(cube).copy(), 1, axes=(1, 0))
    return new_cube


def B_(cube):
    new_cube = cube.copy()
    right(new_cube)[:, 1] = top(cube)[0]
    down(new_cube)[1:2, :] = right(cube)[:, 1][::-1]
    left(new_cube)[:, 0] = down(cube)[1]
    top(new_cube)[0:1, :] = left(cube)[:, 0][::-1]
    back(new_cube)[:] = np.rot90(back(cube).copy(), 1, axes=(0, 1))
    return new_cube


def L(cube):
    new_cube = cube.copy()
    front(new_cube)[:, 0] = top(cube)[:, 0]
    down(new_cube)[:, 0] = front(cube)[:, 0]
    left(new_cube)[:] = np.rot90(left(cube), 1, axes=(1, 0))
    top(new_cube)[:, 0] = back(cube)[:, 1][::-1]
    back(new_cube)[:, 1] = down(cube)[:, 0][::-1]
    return new_cube


def L_(cube):
    new_cube = cube.copy()
    front(new_cube)[:, 0] = down(cube)[:, 0]
    down(new_cube)[:, 0] = back(cube)[:, 1][::-1]
    left(new_cube)[:] = np.rot90(left(cube), 1, axes=(0, 1))
    top(new_cube)[:, 0] = front(cube)[:, 0]
    back(new_cube)[:, 1] = top(cube)[:, 0][::-1]
    return new_cube


def D(cube):
    new_cube = cube.copy()
    front(new_cube)[1:2] = left(cube)[1]
    right(new_cube)[1:2] = front(cube)[1:2]
    down(new_cube)[:] = np.rot90(down(cube), 1, axes=(1, 0))
    left(new_cube)[1:2] = back(cube)[1]
    back(new_cube)[1:2] = right(cube)[1:2]
    return new_cube


def D_(cube):
    new_cube = cube.copy()
    front(new_cube)[1:2] = right(cube)[1]
    right(new_cube)[1:2] = back(cube)[1]
    down(new_cube)[:] = np.rot90(down(cube), 1, axes=(0, 1))
    left(new_cube)[1:2] = front(cube)[1]
    back(new_cube)[1:2] = left(cube)[1:2]
    return new_cube


# In[8]:


REVERSE_MOVES = {
    B: B_,
    L: L_,
    D: D_,

    B_: B,
    L_: L,
    D_: D,

    "B'": B,
    "L'": L,
    "D'": D,
}

MOVES = {
    'B': B,
    "B'": B_,
    "L": L,
    "L'": L_,
    "D": D,
    "D'": D_,
    "B_": B_,
    "L_": L_,
    "D_": D_,
}


def to_half_turns(scramble):
    """
    Translate the FRU notation of scrambles into our BLD notation,
    also translating half rotations like U2 into equivalent quarter rotations like D D.
    """
    return scramble.replace('F', 'B').replace('R', 'L').replace('U', 'D').replace('L2', 'L L').replace('D2',
                                                                                                       'D D').replace(
        'B2', 'B B')


def scramble(scramble_string):
    """
    Scramble a cube using the given scramble string.
    """
    print(f'Translating scramble: {scramble_string}')
    expanded = scramble_string.replace('R2', 'R R').replace('F2', 'F F').replace('U2', 'U U').replace('L2',
                                                                                                      'L L').replace(
        'B2', 'B B').replace('D2', 'D D')
    print(f'Expanded notation:    {expanded}')
    scramble_string = to_half_turns(scramble_string)
    print(f'Equivalent notation:  {scramble_string}')

    scrambled = fresh_cube()
    for move in scramble_string.split(' '):
        scrambled = MOVES[move](scrambled)

    print('Scrambled:')
    print(display(scrambled))
    return scrambled


def pretty_print(move):
    """A nicer string representation of a move function"""
    return move.__name__.replace('_', "'")


def solve_bfs(scrambled):
    visited = defaultdict(list)
    candidates = [scrambled]

    solved = fresh_cube()
    solved_key = solved.tobytes()

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in [B, L, D, B_, L_, D_]:
                current_state = move(old_state)
                current_state_key = current_state.tobytes()
                old_state_key = old_state.tobytes()

                if current_state_key == solved_key:
                    print('Solution:')
                    solution = visited[old_state_key] + [move]
                    print(' '.join([f'{move.__name__:2}' for move in solution]).replace('_', "'"))
                    for m in solution:
                        scrambled = m(scrambled)
                    print(display(scrambled))
                    print('visited:', len(visited), 'states')
                    return solution

                if current_state_key not in visited:
                    visited[current_state_key] = visited[old_state_key] + [move]
                    new_candidates.append(current_state)

        candidates = new_candidates


def solve_constrained_bfs(scrambled):
    CONSTRAINTS = {
        (B,): [L, L_, D, D_, B],
        (B_,): [L, L_, D, D_, B_],
        (L,): [L, D, D_, B, B_],
        (L_,): [L_, D, D_, B, B_],
        (D,): [L, L_, D, B, B_],
        (D_,): [L, L_, D_, B, B_],

        (B, L,): [L, D, D_, B, B_],
        (B, L_,): [L_, D, D_, B, B_],
        (B, D,): [L, L_, D, B, B_],
        (B, D_,): [L, L_, D_, B, B_],

        (L, B,): [L, L_, D, D_, B],
        (L, B_,): [L, L_, D, D_, B_],
        (L, D,): [L, L_, D, B, B_],
        (L, D_,): [L, L_, D_, B, B_],

        (D, B,): [L, L_, D, D_, B],
        (D, B_,): [L, L_, D, D_, B_],
        (D, L,): [L, D, D_, B, B_],
        (D, L_,): [L_, D, D_, B, B_],

        (B_, L,): [L, D, D_, B, B_],
        (B_, L_,): [L_, D, D_, B, B_],
        (B_, D,): [L, L_, D, B, B_],
        (B_, D_,): [L, L_, D_, B, B_],

        (L_, B,): [L, L_, D, D_, B],
        (L_, B_,): [L, L_, D, D_, B_],
        (L_, D,): [L, L_, D, B, B_],
        (L_, D_,): [L, L_, D_, B, B_],

        (D_, B,): [L, L_, D, D_, B],
        (D_, B_,): [L, L_, D, D_, B_],
        (D_, L,): [L, D, D_, B, B_],
        (D_, L_,): [L_, D, D_, B, B_],

        (B, B): [L, L_, D, D_],
        (B_, B_): [L, L_, D, D_],
        (L, L): [B, B_, D, D_],
        (L_, L_): [B, B_, D, D_],
        (D, D): [L, L_, B, B_],
        (D_, D_): [L, L_, B, B_],
        (): [B, L, D, B_, L_, D_],
    }
    visited = defaultdict(list)
    candidates = [scrambled]

    solved = fresh_cube()
    solved_key = solved.tobytes()

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            path = visited.get(old_state.tobytes(), [])
            moves = CONSTRAINTS[tuple(path[-2:])]
            for move in moves:
                old_state = old_state.copy()
                current_state = move(old_state)
                current_state_key = current_state.tobytes()
                old_state_key = old_state.tobytes()

                if current_state_key == solved_key:
                    print('Solution:')
                    solution = visited[old_state_key] + [move]
                    print(' '.join([f'{move.__name__:2}' for move in solution]).replace('_', "'"))
                    for m in solution:
                        scrambled = m(scrambled)
                    print(display(scrambled))
                    print('visited:', len(visited), 'states')

                    return solution

                if current_state_key not in visited:
                    visited[current_state_key] = visited[old_state_key] + [move]
                    new_candidates.append(current_state)

        candidates = new_candidates




def solve_more_constrained_bfs(scrambled):
    CONSTRAINTS = {
        (B,): [L, L_, D, D_, B],
        (B_,): [L, L_, D, D_],
        (L,): [L, D, D_, B, B_],
        (L_,): [D, D_, B, B_],
        (D,): [L, L_, D, B, B_],
        (D_,): [L, L_, B, B_],

        (B, L,): [L, D, D_, B, B_],
        (B, L_,): [D, D_, B, B_],
        (B, D,): [L, L_, D, B, B_],
        (B, D_,): [L, L_, B, B_],

        (L, B,): [L, L_, D, D_, B],
        (L, B_,): [L, L_, D, D_],
        (L, D,): [L, L_, D, B, B_],
        (L, D_,): [L, L_, B, B_],

        (D, B,): [L, L_, D, D_, B],
        (D, B_,): [L, L_, D, D_],
        (D, L,): [L, D, D_, B, B_],
        (D, L_,): [D, D_, B, B_],

        (B_, L,): [L, D, D_, B, B_],
        (B_, L_,): [D, D_, B, B_],
        (B_, D,): [L, L_, D, B, B_],
        (B_, D_,): [L, L_, B, B_],

        (L_, B,): [L, L_, D, D_, B],
        (L_, B_,): [L, L_, D, D_],
        (L_, D,): [L, L_, D, B, B_],
        (L_, D_,): [L, L_, B, B_],

        (D_, B,): [L, L_, D, D_, B],
        (D_, B_,): [L, L_, D, D_],
        (D_, L,): [L, D, D_, B, B_],
        (D_, L_,): [D, D_, B, B_],

        (B, B): [L, L_, D, D_],
        (B_, B_): [L, L_, D, D_],
        (L, L): [B, B_, D, D_],
        (L_, L_): [B, B_, D, D_],
        (D, D): [L, L_, B, B_],
        (D_, D_): [L, L_, B, B_],
        (): [B, L, D, B_, L_, D_],
    }

    visited = defaultdict(list)
    candidates = [scrambled]

    solved = fresh_cube()
    solved_key = solved.tobytes()

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            path = visited.get(old_state.tobytes(), [])
            moves = CONSTRAINTS[tuple(path[-2:])]
            for move in moves:
                old_state = old_state.copy()
                current_state = move(old_state)
                current_state_key = current_state.tobytes()
                old_state_key = old_state.tobytes()

                if current_state_key == solved_key:
                    print('Solution:')
                    solution = visited[old_state_key] + [move]
                    print(' '.join([f'{move.__name__:2}' for move in solution]).replace('_', "'"))
                    for m in solution:
                        scrambled = m(scrambled)
                    print(display(scrambled))
                    print('visited:', len(visited), 'states')

                    return solution

                if current_state_key not in visited:
                    visited[current_state_key] = visited[old_state_key] + [move]
                    new_candidates.append(current_state)

        candidates = new_candidates


def unique_configurations():
    VISITED = {fresh_cube().tobytes(): []}
    candidates = [fresh_cube()]
    MOVES = [ B, D, L, B_, D_, L_]

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in MOVES:
                new_state = move(old_state)
                new_state_key = new_state.tobytes()

                if new_state_key not in VISITED:
                    VISITED[new_state_key] = VISITED[old_state.tobytes()] + [move]
                    new_candidates.append(new_state)

        candidates = new_candidates
    return VISITED


def unique_configurations_constrained():
    CONSTRAINTS = {
        (B,): [L, L_, D, D_, B],
        (B_,): [L, L_, D, D_, B_],
        (L,): [L, D, D_, B, B_],
        (L_,): [L_, D, D_, B, B_],
        (D,): [L, L_, D, B, B_],
        (D_,): [L, L_, D_, B, B_],

        (B, L,): [L, D, D_, B, B_],
        (B, L_,): [L_, D, D_, B, B_],
        (B, D,): [L, L_, D, B, B_],
        (B, D_,): [L, L_, D_, B, B_],

        (L, B,): [L, L_, D, D_, B],
        (L, B_,): [L, L_, D, D_, B_],
        (L, D,): [L, L_, D, B, B_],
        (L, D_,): [L, L_, D_, B, B_],

        (D, B,): [L, L_, D, D_, B],
        (D, B_,): [L, L_, D, D_, B_],
        (D, L,): [L, D, D_, B, B_],
        (D, L_,): [L_, D, D_, B, B_],

        (B_, L,): [L, D, D_, B, B_],
        (B_, L_,): [L_, D, D_, B, B_],
        (B_, D,): [L, L_, D, B, B_],
        (B_, D_,): [L, L_, D_, B, B_],

        (L_, B,): [L, L_, D, D_, B],
        (L_, B_,): [L, L_, D, D_, B_],
        (L_, D,): [L, L_, D, B, B_],
        (L_, D_,): [L, L_, D_, B, B_],

        (D_, B,): [L, L_, D, D_, B],
        (D_, B_,): [L, L_, D, D_, B_],
        (D_, L,): [L, D, D_, B, B_],
        (D_, L_,): [L_, D, D_, B, B_],

        (B, B): [L, L_, D, D_],
        (B_, B_): [L, L_, D, D_],
        (L, L): [B, B_, D, D_],
        (L_, L_): [B, B_, D, D_],
        (D, D): [L, L_, B, B_],
        (D_, D_): [L, L_, B, B_],
        (): [B, L, D, B_, L_, D_],
    }

    VISITED = {fresh_cube().tobytes(): []}
    candidates = [fresh_cube()]

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in CONSTRAINTS[tuple(VISITED[old_state.tobytes()])[-2:]]:
                new_state = move(old_state)
                new_state_key = new_state.tobytes()

                if new_state_key not in VISITED:
                    VISITED[new_state_key] = VISITED[old_state.tobytes()] + [move]
                    new_candidates.append(new_state)

        candidates = new_candidates
    return VISITED

def unique_configurations_more_constrained():
    CONSTRAINTS = {
        (B,): [L, L_, D, D_, B],
        (B_,): [L, L_, D, D_],
        (L,): [L, D, D_, B, B_],
        (L_,): [D, D_, B, B_],
        (D,): [L, L_, D, B, B_],
        (D_,): [L, L_, B, B_],

        (B, L,): [L, D, D_, B, B_],
        (B, L_,): [D, D_, B, B_],
        (B, D,): [L, L_, D, B, B_],
        (B, D_,): [L, L_, B, B_],

        (L, B,): [L, L_, D, D_, B],
        (L, B_,): [L, L_, D, D_],
        (L, D,): [L, L_, D, B, B_],
        (L, D_,): [L, L_, B, B_],

        (D, B,): [L, L_, D, D_, B],
        (D, B_,): [L, L_, D, D_],
        (D, L,): [L, D, D_, B, B_],
        (D, L_,): [D, D_, B, B_],

        (B_, L,): [L, D, D_, B, B_],
        (B_, L_,): [D, D_, B, B_],
        (B_, D,): [L, L_, D, B, B_],
        (B_, D_,): [L, L_, B, B_],

        (L_, B,): [L, L_, D, D_, B],
        (L_, B_,): [L, L_, D, D_],
        (L_, D,): [L, L_, D, B, B_],
        (L_, D_,): [L, L_, B, B_],

        (D_, B,): [L, L_, D, D_, B],
        (D_, B_,): [L, L_, D, D_],
        (D_, L,): [L, D, D_, B, B_],
        (D_, L_,): [D, D_, B, B_],

        (B, B): [L, L_, D, D_],
        (B_, B_): [L, L_, D, D_],
        (L, L): [B, B_, D, D_],
        (L_, L_): [B, B_, D, D_],
        (D, D): [L, L_, B, B_],
        (D_, D_): [L, L_, B, B_],
        (): [B, L, D, B_, L_, D_],
    }

    VISITED = {fresh_cube().tobytes(): []}
    candidates = [fresh_cube()]

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in CONSTRAINTS[tuple(VISITED[old_state.tobytes()])[-2:]]:
                new_state = move(old_state)
                new_state_key = new_state.tobytes()

                if new_state_key not in VISITED:
                    VISITED[new_state_key] = VISITED[old_state.tobytes()] + [move]
                    new_candidates.append(new_state)

        candidates = new_candidates
    return VISITED
