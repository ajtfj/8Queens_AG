fenotype = [5, 2, 0, 6, 4, 7, 1, 3]

board = [['□' for _ in range(8)] for _ in range(8)]
for col, row in enumerate(fenotype):
    board[row][col] = "■"

for row in board:
    for elem in row:
        print(elem, end=" ")
    print()