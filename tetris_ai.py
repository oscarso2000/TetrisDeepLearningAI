from collections import defaultdict, OrderedDict
import pygame, numpy, time, threading, random, sys
from logs import CustomTensorBoard
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm

# Rotates a shape clockwise
def rotate_clockwise(shape):
  return [ [ shape[y][x]
      for y in range(len(shape)) ]
    for x in range(len(shape[0]) - 1, -1, -1) ]

# checks if there is a collision in any direction
def check_collision(board, shape, offset):
  off_x, off_y = offset
  for cy, row in enumerate(shape):
    for cx, cell in enumerate(row):
      try:
        if cell and board[ cy + off_y ][ cx + off_x ]:
          return True
      except IndexError:
        return True
  return False

# Used for adding a tetromino to the board
def join_matrixes(mat1, mat2, mat2_off):
  off_x, off_y = mat2_off
  for cy, row in enumerate(mat2):
    for cx, val in enumerate(row):
      try:
        mat1[cy+off_y-1  ][cx+off_x] += val
      except IndexError:
        print("out of bounds join")
  return mat1

'''
  TetrisAI
'''
class TetrisAI(object):
  
  def __init__(self, tetris_app, agent, step_size, model_name, logs_file):
    self.name = "Crehg"
    self.tetris_app = tetris_app
    self.agent = agent
    self.model_name = 'trained_models/' + model_name
    self.logs_file = logs_file
    self.step_size = step_size
    self.screen = pygame.display.set_mode((200, 480))

    self.tetris_shapes = [
        [[1, 1, 1],
        [0, 1, 0]],
        
        [[0, 2, 2],
        [2, 2, 0]],
        
        [[3, 3, 0],
        [0, 3, 3]],
        
        [[4, 0, 0],
        [4, 4, 4]],
        
        [[0, 0, 5],
        [5, 5, 5]],
        
        [[6, 6, 6, 6]],
        
        [[7, 7],
        [7, 7]]
      ]

  def draw_matrix(self, matrix, offset, color=(255,255,255)):
    off_x, off_y  = offset
    for y, row in enumerate(matrix):
      for x, val in enumerate(row):
        if val:
          pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(
              (off_x+x) *
                20,
              (off_y+y) *
                20, 
              20,
              20),0)

  '''
    Setters
  '''
  def set_board(self, board):
    self.board = board

  def set_tetromino(self, tetromino, tetromino_x, tetromino_y):
    self.tetromino = tetromino
    self.tetromino_x = tetromino_x
    self.tetromino_y = tetromino_y

  '''
    Actual AI stuff
  '''
  # move a piece horizontally
  def move(self, desired_x, board, tetromino, tetromino_x, tetromino_y):

    while(tetromino_x != desired_x):
      dist = desired_x - tetromino_x
      delta_x = int(dist/abs(dist))

      new_x = tetromino_x + delta_x
      if not check_collision(board,
                             tetromino,
                             (new_x, tetromino_y)):
        tetromino_x = new_x
      else:
        break
    return tetromino_x
 
  '''
    Rotate tetromino if no collision
  ''' 
  def rotate_tetromino(self, board, tetromino, tetromino_x, tetromino_y):

    new_tetromino = rotate_clockwise(tetromino)
    if not check_collision(board,
                           new_tetromino,
                           (tetromino_x, tetromino_y)):
      return new_tetromino
    return tetromino

  '''
    Try moving piece down
      - if collision:
        1. add tetromino to board
        2. Check for row completion
        3. if no collision drop again
  ''' 
  def drop(self, board, tetromino, tetromino_x, tetromino_y):

    tetromino_y += 1
    if check_collision(board,
                       tetromino,
                       (tetromino_x, tetromino_y)):
      board = join_matrixes(
        board,
        tetromino,
        (tetromino_x, tetromino_y))
    else:
      self.drop(board, tetromino, tetromino_x, tetromino_y)
    return board, tetromino_y
  
  '''
    Gets the height of each column
  '''
  def get_column_heights(self, board):
    # get the hights of each column
    heghts = [0 for i in board[0]] 

    for y, row in enumerate(board[::-1]):
      for x, val in enumerate(row):
        if val != 0:
          heghts[x] = y
    
    return heghts

  '''
    Find max height in board
  '''
  def get_max_height(self, board):
    return max(self.get_column_heights(board))

  '''
    Gets the sum of all the columns
  '''
  def get_cumulative_height(self, board):
    return sum(self.get_column_heights(board))

  '''
    Gets the difference betweent he shortest and tallest height
  '''
  def get_relative_height(self, board):
    column_heights= self.get_column_heights(board)
    max_height = max(column_heights)
    min_height = min(column_heights)
    return max_height - min_height

  '''
    Get roughness
      determined by summing the hight
      absolute difference between a row at i and i+1
  '''
  def get_roughness(self, board):
    levels = self.get_column_heights(board)
    roughness = 0
    
    for x in range(len(levels)-1):
      roughness += abs(levels[x] - levels[x+1]) 
    return roughness

  '''
    Get's the number of spaces which are un reacable
      A space is un reachable if there is another piece above it
      even if you could slip the piece in from the side
  '''
  def get_hole_count(self, board):
    levels = self.get_column_heights(board) 
    holes = 0

    for y, row in enumerate(board[::-1]):
      for x, val in enumerate(row):
        # if below max column height and is a zero
        if y < levels[x] and val==0:
          holes += 1 
    return holes

  '''
    Check how many rows will be cleared in this config
  '''
  def get_rows_cleared(self, board):
    # starts at -1 to account for bottom row which
    # is always all 1
    rows_cleared = -1
    
    for row in board:
      if 0 not in row:
        rows_cleared += 1
    return rows_cleared 

  '''
    Check how many tiles are touching the floor
  '''
  def get_floor_blocks(self,board):
    floor = board[23]
    counter = 0
    for i in floor:
      if i != 0:
        counter +=1
    return counter

  '''
    Check how many tiles are touching the wall
  '''
  def get_wall_blocks(self, board):
    wall_blocks = 0
    for index, row in enumerate(board):
      if row[0] != 0:
        wall_blocks += 1
      if row[-1] != 0:
        wall_blocks +=1
    return wall_blocks

  '''
    Check how many contiguous sections there are using DFS
  '''
  def get_contig_sections(self, board):
    # visited = [[False for _ in range(len(board[0]))] for _ in range(len(board))]
    visited = []
    counter = 0
    for i in range(len(board)):
      visited.append([True if board[i][j]==0 else False for j in range(len(board[i]))])
    for i, row in enumerate(board):
      for j, element in enumerate(row):
        if visited[i][j] == False:
          stack = []
          stack.append((i,j))
          while (len(stack)):
            si, sj = stack[-1]
            stack.pop()
            if (not visited[si][sj]):
              visited[si][sj] = True
            # for node in self.adj[s]:
            #Down
            if si != len(board)-1:
              if(not visited[si+1][sj]):
                stack.append((si+1,sj))
            #Right
            if sj != len(board[si])-1:
              if(not visited[si][sj+1]):
                stack.append((si,sj+1))
            #Up
            if si != 0:
              if(not visited[si-1][sj]):
                stack.append((si-1,sj))
            #Left
            if sj != 0:
              if(not visited[si][sj-1]):
                stack.append((si,sj-1))
          counter += 1
    return counter

  '''
    Deep Q Learning Code 
  '''
  def quit(self):
        self.tetris_app.quit()

  def _get_board_props(self):
    cum_height = self.get_cumulative_height(self.board)
    rows_cleared = self.get_rows_cleared(self.board)
    hole_count = self.get_hole_count(self.board)
    rough = self.get_roughness(self.board)
    floor_blocks = self.get_floor_blocks(self.board)
    contig_sections = self.get_contig_sections(self.board)
    return [rows_cleared, hole_count, rough, cum_height, floor_blocks, contig_sections]

  def get_next_states(self):
    states = {}
    if not (hasattr(self, "board") and hasattr(self, "tetromino")):
      raise ValueError("either board or tetromino do not exist for TetrisAI")
    
    cur_state = self.tetris_app.get_state()

    self.set_board(cur_state["board"])
    self.set_tetromino(cur_state["tetromino"], cur_state["tetromino_x"], cur_state["tetromino_y"])

    temp_board = numpy.copy(self.board)
    temp_tetromino = numpy.copy(self.tetromino)
    tetromino_number = self.get_tetromino_number(temp_tetromino)
    temper = numpy.copy(temp_tetromino)

    temp_x = self.tetromino_x
    temp_y = self.tetromino_y

    # contains all the board orientations possible with the current tetromino
    boards = []
    rotation = 0
    rot = 0
    
    if temp_tetromino[0][0] == 7:
      num_rot = 1
    elif temp_tetromino[0][0] == 6:
      num_rot = 2
    else:
      num_rot = 4

    for j in range(num_rot):
      for i in range(len(self.board[0])):
        temp_x = self.move(i, temp_board, temp_tetromino, temp_x, temp_y)
        temp_board, temp_y = self.drop(temp_board, temp_tetromino, temp_x, temp_y)

        cum_height = self.get_cumulative_height(temp_board)
        rows_cleared = self.get_rows_cleared(temp_board)
        hole_count = self.get_hole_count(temp_board)
        rough = self.get_roughness(temp_board)
        floor_blocks = self.get_floor_blocks(temp_board)
        contig_sections = self.get_contig_sections(temp_board)
        boards.append(temp_board)
        states[(i, rotation)] = [rows_cleared, hole_count, rough, cum_height, floor_blocks, contig_sections]
        
        temp_board = numpy.copy(self.board)
        temp_x = self.tetromino_x
        temp_y = self.tetromino_y

      temp_tetromino = self.rotate_tetromino(temp_board, temp_tetromino, temp_x, temp_y)
      rotation += 90
      rot += 1
    return states


  def get_tetromino_number(self, tetromino):
        if tetromino[0][0] == 6:
          return 5
        if tetromino[0][0] == 1:
          return 0
        if tetromino[0][1] == 2:
          return 1
        if tetromino[0][0] == 3:
          return 2
        if tetromino[0][0] == 4:
          return 3
        if tetromino[0][0] == 7:
          return 6
        if tetromino[1][0] == 5:
          return 4
        else:
          return -1
        
  def get_actions(self, best_action):
    actions = []

    # CHANGE THIS LATER
    # best_score = scores.index( max(scores) )
    rot = best_action[1] // 90

    # rotate to proper orientation
    # rotations = [ "up" for i in range( best_score//len(self.board[0]) )]
    rotations = ["up" for i in range(rot)]
    actions.extend(rotations)

    # move to proper x pos
    desired_x = best_action[0]
    wanted_move = ""  
    if( desired_x > self.tetromino_x ):
      wanted_move = "right" 
    elif(desired_x < self.tetromino_x):
      wanted_move = "left"
    
    moves = [ wanted_move for i in range( abs(desired_x - self.tetromino_x) ) ]
    actions.extend(moves)

    # move to proper y pos
    levels =  self.get_column_heights(self.board)
    desired_y = levels[desired_x] + len(self.tetromino) 

    world_y = len(self.board) - self.tetromino_y
    num_drops = world_y - desired_y
    drops = [ "down" for i in range( num_drops ) ]
    #actions.extend(drops)

    return actions

  '''
  Create the dql algorithm
  '''

  def load_board(self):
    cur_state = self.tetris_app.get_state()
    self.set_board(cur_state["board"])
    self.set_tetromino(cur_state["tetromino"], cur_state["tetromino_x"], cur_state["tetromino_y"])
    return cur_state

  def start_dql(self, training = True, episodes = 5000):
    pbar = tqdm(total = 10000+1)
    n_neurons = [32, 32]
    mem_size = 20000
    batch_size = 512
    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={1}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)
    episode = 1
    train_every = self.step_size
    scores = []
    count_rows_cleared = 0
    while training:

      cur_state = self.tetris_app.get_state()
      self.set_board(cur_state["board"])
      self.set_tetromino(cur_state["tetromino"], cur_state["tetromino_x"], cur_state["tetromino_y"])
  
      if not cur_state["needs_actions"]:
        continue
      actions = []

      # When game ends 
      if cur_state["gameover"]:
        scores.append(self.tetris_app.score)
        actions.append("space")
        self.tetris_app.add_actions(actions)
        
        if episode % 20 == 0:
        # Logs
          avg_score = mean(scores[-5:])
          min_score = min(scores[-5:])
          max_score = max(scores[-5:])
          # Add TIME?
          log._add(episode, avg_score, max_score, min_score, self.logs_file)
          log.log(episode, avg_score, max_score, min_score)

        # Train Model  
        if episode % train_every == 0: 
          self.agent.train(batch_size=512, epochs=1)

        # Save model
        if episode % 50 == 0:
          self.agent.model.save(self.model_name)
          print(" MEAN SCORES: ", mean(scores[-20:]))
          print("Epsilon: ", self.agent.epsilon)
          # print("Episode: ", episode)
          # print("Total Rows Cleared: ", count_rows_cleared)
          # print("------------------------------")

        # Update
        time.sleep(0.5)
        episode += 1
        pbar.update(1)

      # if not done 

      try:
        current_state = self._get_board_props()
        next_states = self.get_next_states()
        best_state, reward = self.agent.best_state(next_states)
        best_action = None 
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        actions = self.get_actions(best_action)
        self.tetris_app.add_actions(actions)
        # Update
        self.agent.add_to_memory(current_state, next_states[best_action], reward, cur_state["gameover"])



      # count_rows_cleared += current_state[0]
      except:
        print("Didnt fucking work...")
        print("Best State + Reward : ", best_state, reward)
        print("Best Action: ", best_action)
        print("Current State", cur_state["board"])
        print("-----------------------------------")


  def test_dql(self, testing = True):
    scores = []
    while testing:
      cur_state = self.load_board()
      cur_state = self.tetris_app.get_state()

      self.set_board(cur_state["board"])
      self.set_tetromino(cur_state["tetromino"], cur_state["tetromino_x"], cur_state["tetromino_y"])
  
      if not cur_state["needs_actions"]:
        continue
      actions = []

      # When game ends 
      if cur_state["gameover"]:
        scores.append(self.tetris_app.score)
        # print("GAMEOVER", len(self.agent.memory), self.tetris_app.score)
        actions.append("space")
        self.tetris_app.add_actions(actions)
        # print("Episode:", episode, "SCORE", self.tetris_app.score, len(self.agent.memory))
        time.sleep(0.5)

      current_state = self._get_board_props()
      next_states = self.get_next_states()
      best_state, reward = self.agent.best_state(next_states)
      best_action = None 
      for action, state in next_states.items():
          if state == best_state:
              best_action = action
              break
      actions = self.get_actions(best_action)
      self.tetris_app.add_actions(actions)