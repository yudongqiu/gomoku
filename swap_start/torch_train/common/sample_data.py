import sys
from gomoku_train_swap import load_data_h5

def get_state(x1):
    is_black = bool(x1[2,0,0])
    return (x1[0] - x1[1]) if is_black else (x1[1] - x1[0])

def show_state(state):
    board_size = 15
    print(' '*4 + ' '.join([chr(97+i) for i in range(board_size)]))
    print(' '*3 + '='*(2*board_size))
    for x in range(board_size):
        row = ['%2s|'%x]
        for y in range(board_size):
            if state[x,y] == 1:
                c = 'x'
            elif state[x,y] == -1:
                c = 'o'
            else:
                c = '-'
            row.append(c)
        print(' '.join(row))

train_X, train_Y, train_W = load_data_h5(sys.argv[1])
print(f'Loaded {len(train_X)} data')

for i in range(100):
    
    x = train_X[i]
    y = train_Y[i,0]
    if abs(y) > 0.4:
        # print(x)
        player = 'black' if bool(x[2,0,0]) else 'white'
        state = get_state(x)
        show_state(state)
        print(f"{player} Y = {y}")
