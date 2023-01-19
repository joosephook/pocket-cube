import numpy as np
from collections import defaultdict

def B (cube, idx=np.array([0, 1, 2, 17, 4, 5, 9, 7, 3, 15, 22, 10, 12, 13, 14, 16, 6, 18, 8, 19, 20, 21, 23, 11], dtype=np.int64)): return cube[idx]
def D (cube, idx=np.array([0, 1, 2, 3, 16, 4, 6, 7, 8, 9, 10, 11, 18, 19, 12, 13, 17, 5, 22, 23, 20, 21, 14, 15], dtype=np.int64)): return cube[idx]
def L (cube, idx=np.array([8, 1, 2, 3, 0, 5, 18, 6, 23, 9, 10, 16, 20, 13, 14, 15, 12, 17, 19, 7, 11, 21, 22, 4], dtype=np.int64)): return cube[idx]
def B_(cube, idx=np.array([0, 1, 2, 8, 4, 5, 16, 7, 18, 6, 11, 23, 12, 13, 14, 9, 15, 3, 17, 19, 20, 21, 10, 22], dtype=np.int64)): return cube[idx]
def D_(cube, idx=np.array([0, 1, 2, 3, 5, 17, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 4, 16, 12, 13, 20, 21, 18, 19], dtype=np.int64)): return cube[idx]
def L_(cube, idx=np.array([4, 1, 2, 3, 23, 5, 7, 19, 0, 9, 10, 20, 16, 13, 14, 15, 11, 17, 6, 18, 12, 21, 22, 8], dtype=np.int64)): return cube[idx]

def fresh_cube():
    return np.arange(24, dtype=np.uint8)+1


EMPTY = {0: '  '}
BLACK = {num: "\U00002B1B" for num in [1, 2, 13, 14]}
RED = {num: "\U0001F7E5" for num in [3, 4, 15, 16]}
GREEN = {num: "\U0001F7E9" for num in [5, 6, 17, 18]}
ORANGE = {num: "\U0001F7E7" for num in [7, 8, 19, 20]}
BLUE = {num: "\U0001F7E6" for num in [9, 10, 21, 22]}
YELLOW = {num: "\U0001F7E8" for num in [11, 12, 23, 24]}

UNICODE = EMPTY | BLACK | RED | GREEN | ORANGE | BLUE | YELLOW

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


def solve_array_bfs(scrambled):
    B = [0, 1, 2, 17, 4, 5, 9, 7, 3, 15, 22, 10, 12, 13, 14, 16, 6, 18, 8, 19, 20, 21, 23, 11]
    D = [0, 1, 2, 3, 16, 4, 6, 7, 8, 9, 10, 11, 18, 19, 12, 13, 17, 5, 22, 23, 20, 21, 14, 15]
    L = [8, 1, 2, 3, 0, 5, 18, 6, 23, 9, 10, 16, 20, 13, 14, 15, 12, 17, 19, 7, 11, 21, 22, 4]
    B_ = [0, 1, 2, 8, 4, 5, 16, 7, 18, 6, 11, 23, 12, 13, 14, 9, 15, 3, 17, 19, 20, 21, 10, 22]
    D_ = [0, 1, 2, 3, 5, 17, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 4, 16, 12, 13, 20, 21, 18, 19]
    L_ = [4, 1, 2, 3, 23, 5, 7, 19, 0, 9, 10, 20, 16, 13, 14, 15, 11, 17, 6, 18, 12, 21, 22, 8]

    B = np.array(B).astype(np.int64)
    D = np.array(D).astype(np.int64)
    L = np.array(L).astype(np.int64)
    B_ = np.array(B_).astype(np.int64)
    D_ = np.array(D_).astype(np.int64)
    L_ = np.array(L_).astype(np.int64)

    visited = defaultdict(list)
    candidates = [scrambled]

    solved = fresh_cube()
    solved_key = solved.tobytes()
    moves = (B, L, D, B_, L_, D_)
    names = ('B', 'L', 'D', 'B\'', 'L\'', 'D\'')
    names = {m.tobytes():n for m,n in zip(moves, names)}

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in moves:
                current_state = old_state[move]
                current_state_key = current_state.tobytes()
                old_state_key = old_state.tobytes()

                if current_state_key == solved_key:
                    print('Solution:')
                    solution = visited[old_state_key] + [move]
                    print(' '.join([names[m.tobytes]] for m in solution))
                    for m in solution:
                        scrambled = scrambled[m]
                    print(display(scrambled))
                    return solution

                if current_state_key not in visited:
                    visited[current_state_key] = visited[old_state_key] + [move]
                    new_candidates.append(current_state)

        candidates = new_candidates


def unique_configurations():
    B = [0, 1, 2, 17, 4, 5, 9, 7, 3, 15, 22, 10, 12, 13, 14, 16, 6, 18, 8, 19, 20, 21, 23, 11]
    D = [0, 1, 2, 3, 16, 4, 6, 7, 8, 9, 10, 11, 18, 19, 12, 13, 17, 5, 22, 23, 20, 21, 14, 15]
    L = [8, 1, 2, 3, 0, 5, 18, 6, 23, 9, 10, 16, 20, 13, 14, 15, 12, 17, 19, 7, 11, 21, 22, 4]
    B_ = [0, 1, 2, 8, 4, 5, 16, 7, 18, 6, 11, 23, 12, 13, 14, 9, 15, 3, 17, 19, 20, 21, 10, 22]
    D_ = [0, 1, 2, 3, 5, 17, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 4, 16, 12, 13, 20, 21, 18, 19]
    L_ = [4, 1, 2, 3, 23, 5, 7, 19, 0, 9, 10, 20, 16, 13, 14, 15, 11, 17, 6, 18, 12, 21, 22, 8]

    B = np.array(B).astype(np.int64)
    D = np.array(D).astype(np.int64)
    L = np.array(L).astype(np.int64)
    B_ = np.array(B_).astype(np.int64)
    D_ = np.array(D_).astype(np.int64)
    L_ = np.array(L_).astype(np.int64)
    MOVES = [ B, D, L, B_, D_, L_]
    VISITED = {fresh_cube().tobytes(): []}
    candidates = [fresh_cube()]

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in MOVES:
                new_state = old_state[move]
                new_state_key = new_state.tobytes()

                if new_state_key not in VISITED:
                    VISITED[new_state_key] = VISITED[old_state.tobytes()] + [move]
                    new_candidates.append(new_state)

        candidates = new_candidates
    return VISITED


def unique_configurations_constrained():
    B = [0, 1, 2, 17, 4, 5, 9, 7, 3, 15, 22, 10, 12, 13, 14, 16, 6, 18, 8, 19, 20, 21, 23, 11]
    D = [0, 1, 2, 3, 16, 4, 6, 7, 8, 9, 10, 11, 18, 19, 12, 13, 17, 5, 22, 23, 20, 21, 14, 15]
    L = [8, 1, 2, 3, 0, 5, 18, 6, 23, 9, 10, 16, 20, 13, 14, 15, 12, 17, 19, 7, 11, 21, 22, 4]
    B_ = [0, 1, 2, 8, 4, 5, 16, 7, 18, 6, 11, 23, 12, 13, 14, 9, 15, 3, 17, 19, 20, 21, 10, 22]
    D_ = [0, 1, 2, 3, 5, 17, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 4, 16, 12, 13, 20, 21, 18, 19]
    L_ = [4, 1, 2, 3, 23, 5, 7, 19, 0, 9, 10, 20, 16, 13, 14, 15, 11, 17, 6, 18, 12, 21, 22, 8]

    B = np.array(B)
    D = np.array(D)
    L = np.array(L)
    B_ = np.array(B_)
    D_ = np.array(D_)
    L_ = np.array(L_)
    MOVES = np.array([B, D, L, B_, D_, L_]).astype(np.int64)

    B, D, L, B_, D_, L_ = np.arange(6)
    print(MOVES.shape)
    assert MOVES.shape == (6, 24)
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

    level = 0
    moves = [B, D, L, B_, D_, L_]

    while len(candidates):
        new_candidates = []
        for old_state in candidates:
            for move in CONSTRAINTS[tuple(VISITED[old_state.tobytes()])[-2:]]:
                new_state = old_state[MOVES[move]]
                new_state_key = new_state.tobytes()

                if new_state_key not in VISITED:
                    VISITED[new_state_key] = VISITED[old_state.tobytes()] + [move]
                    new_candidates.append(new_state)

        candidates = new_candidates
    return VISITED

def unique_configurations_more_constrained():
    B = [0, 1, 2, 17, 4, 5, 9, 7, 3, 15, 22, 10, 12, 13, 14, 16, 6, 18, 8, 19, 20, 21, 23, 11]
    D = [0, 1, 2, 3, 16, 4, 6, 7, 8, 9, 10, 11, 18, 19, 12, 13, 17, 5, 22, 23, 20, 21, 14, 15]
    L = [8, 1, 2, 3, 0, 5, 18, 6, 23, 9, 10, 16, 20, 13, 14, 15, 12, 17, 19, 7, 11, 21, 22, 4]
    B_ = [0, 1, 2, 8, 4, 5, 16, 7, 18, 6, 11, 23, 12, 13, 14, 9, 15, 3, 17, 19, 20, 21, 10, 22]
    D_ = [0, 1, 2, 3, 5, 17, 6, 7, 8, 9, 10, 11, 14, 15, 22, 23, 4, 16, 12, 13, 20, 21, 18, 19]
    L_ = [4, 1, 2, 3, 23, 5, 7, 19, 0, 9, 10, 20, 16, 13, 14, 15, 11, 17, 6, 18, 12, 21, 22, 8]

    B = np.array(B).astype(np.int64)
    D = np.array(D).astype(np.int64)
    L = np.array(L).astype(np.int64)
    B_ = np.array(B_).astype(np.int64)
    D_ = np.array(D_).astype(np.int64)
    L_ = np.array(L_).astype(np.int64)

    MOVES = np.array([B, D, L, B_, D_, L_])

    B, D, L, B_, D_, L_ = np.arange(6)

    assert MOVES.shape == (6, 24)

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
                new_state = old_state[MOVES[move]]
                new_state_key = new_state.tobytes()

                if new_state_key not in VISITED:
                    VISITED[new_state_key] = VISITED[old_state.tobytes()] + [move]
                    new_candidates.append(new_state)

        candidates = new_candidates
    return VISITED