#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Copyright (c) 2010 "Kevin Chabowski"<kevin@kch42.de>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

'''
This script is used to quickly train and get genes to then apply to the normal tetris
'''

from random import randrange as rand
import pygame, sys, numpy, math, time
# The configuration
config = {
  'cell_size':  20,
  'cols':    10,
  'rows':    24,
  'delay':  50,
  'maxfps':  math.inf
}

colors = [
(0,   0,   0  ),
(255, 0,   0  ),
(0,   150, 0  ),
(0,   0,   255),
(255, 120, 0  ),
(255, 255, 0  ),
(180, 0,   255),
(0,   220, 220)
]

# Define the shapes of the single parts
tetris_shapes = [
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

# clearing a row for getting points
def remove_row(board, row):
  del board[row]
  return [[0 for i in range(config['cols'])]] + board

# Used for adding a tetromino to the board
def join_matrixes(mat1, mat2, mat2_off):
  off_x, off_y = mat2_off
  for cy, row in enumerate(mat2):
    for cx, val in enumerate(row):
      mat1[cy+off_y-1  ][cx+off_x] += val
  return mat1

# new game
def new_board():
  board = [ [ 0 for x in range(config['cols']) ]
      for y in range(config['rows']) ]
  board += [[ 1 for x in range(config['cols'])]]
  return board

class TetrisApp(object):
  def __init__(self):
    pygame.init()
    pygame.key.set_repeat(250,25)
    self.width = config['cell_size']*config['cols']
    self.height = config['cell_size']*config['rows']
    
    self.screen = pygame.display.set_mode((self.width, self.height))
    pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
                                                 # mouse movement
                                                 # events, so we
                                                 # block them.
    self.actions = []
    self.needs_actions = True
    self.init_game()

  '''
  Creates random shape
  places shape at the top of the boar din the middle
  checks if collision - if yes game over
  '''  
  def new_tetromino(self):
    # every tetromino increases score (fitness)
    self.score += 1

    self.tetromino = tetris_shapes[rand(len(tetris_shapes))]
    self.tetromino_x = int(config['cols'] / 2 - len(self.tetromino[0])/2)
    self.tetromino_y = 0
    
    if check_collision(self.board,
                       self.tetromino,
                       (self.tetromino_x, self.tetromino_y)):
      self.gameover = True

  '''
    Starts a new game
      - new board
      - new tetromino
  '''  
  def init_game(self):
    self.score = 0
    self.board = new_board()
    self.new_tetromino()

    return

  # Show given message at the center of the board (for losing)
  def center_msg(self, msg):
    for i, line in enumerate(msg.splitlines()):
      msg_image =  pygame.font.Font(
        pygame.font.get_default_font(), 12).render(
          line, False, (255,255,255), (0,0,0))
    
      msgim_center_x, msgim_center_y = msg_image.get_size()
      msgim_center_x //= 2
      msgim_center_y //= 2
    
      self.screen.blit(msg_image, (
        self.width // 2-msgim_center_x,
        self.height // 2-msgim_center_y+i*22))

  # Show given message at the center of the board (for losing)
  def show_score(self, msg):
    for i, line in enumerate(msg.splitlines()):
      msg_image =  pygame.font.Font(
        pygame.font.get_default_font(), 12).render(
          line, False, (255,255,255), (0,0,0))
    
      msgim_center_x, msgim_center_y = msg_image.get_size()
      msgim_center_x //= 2
      msgim_center_y //= 2
    
      self.screen.blit(msg_image, (
        (self.width/10),
        (self.height)/10))

  '''
    draws a given matrix
      - used for both the board and the piece
  '''
  def draw_matrix(self, matrix, offset):
    off_x, off_y  = offset
    for y, row in enumerate(matrix):
      for x, val in enumerate(row):
        if val:
          pygame.draw.rect(
            self.screen,
            colors[val],
            pygame.Rect(
              (off_x+x) *
                config['cell_size'],
              (off_y+y) *
                config['cell_size'], 
              config['cell_size'],
              config['cell_size']),0)
    self.show_score(str(self.score))

  # move a piece horizontally
  def move(self, delta_x):
    if not self.gameover and not self.paused:
      new_x = self.tetromino_x + delta_x
      if new_x < 0:
        new_x = 0
      if new_x > config['cols'] - len(self.tetromino[0]):
        new_x = config['cols'] - len(self.tetromino[0])
      if not check_collision(self.board,
                             self.tetromino,
                             (new_x, self.tetromino_y)):
        self.tetromino_x = new_x


  # quit game
  def quit(self):
    self.center_msg("Exiting...")
    pygame.display.update()
    sys.exit()

  '''
    Try moving piece down
      - if collision:
        1. add tetromino to board
        2. create new piece
        3. Check for row completion
  ''' 
  def drop(self):
    if not self.gameover and not self.paused:
      while True:
        self.tetromino_y += 1
        if check_collision(self.board,
                           self.tetromino,
                           (self.tetromino_x, self.tetromino_y)):
          self.board = join_matrixes(
            self.board,
            self.tetromino,
            (self.tetromino_x, self.tetromino_y))
            
          self.new_tetromino()
          self.needs_actions = True
          break
      while True:
        for i, row in enumerate(self.board[:-1]):
          if 0 not in row:
            self.score += 10
            self.board = remove_row(
              self.board, i)
            break
        else:
          break
 
  '''
    Rotate tetromino if no collision
  ''' 
  def rotate_tetromino(self):
    if not self.gameover and not self.paused:
      new_tetromino = rotate_clockwise(self.tetromino)
      if not check_collision(self.board,
                             new_tetromino,
                             (self.tetromino_x, self.tetromino_y)):
        self.tetromino = new_tetromino
 
  # pause game 
  def toggle_pause(self):
    self.paused = not self.paused
 
  # start new game if gameover 
  def start_game(self):
    if self.gameover:
      self.init_game()
      self.gameover = False
 
  '''
    Runs the game
    setup game mechanics
  ''' 
  def run(self):
    # controls
    key_actions = {
      'ESCAPE':  self.quit,
      'LEFT':    lambda:self.move(-1),
      'RIGHT':  lambda:self.move(+1),
      'DOWN':    self.drop,
      'UP':    self.rotate_tetromino,
      'p':    self.toggle_pause,
      'SPACE':  self.start_game
    }
    
    self.gameover = False
    self.paused = False
   
    #Timer to handle drop
    pygame.time.set_timer(pygame.USEREVENT+1, config['delay'])
    dont_burn_my_cpu = pygame.time.Clock()
    while 1:
      self.screen.fill((0,0,0))
      if self.gameover:
        self.center_msg("""Game Over!
Press space to continue""")
        self.needs_actions = True
      else:
        if self.paused:
          self.center_msg("Paused")
        else:
          self.draw_matrix(self.board, (0,0))
          self.draw_matrix(self.tetromino,
                           (self.tetromino_x,
                            self.tetromino_y))
      pygame.display.update()
     
      # check for events 

      '''for event in pygame.event.get():
        #if event.type == pygame.USEREVENT+1:
          #self.drop()
        if event.type == pygame.QUIT:
          self.quit()
        elif event.type == pygame.KEYDOWN:
          for key in key_actions:
            if event.key == eval("pygame.K_"
            +key):
              key_actions[key]()'''


      for action in self.actions:
        action = action.upper()
        if action in key_actions:
          key_actions[action]()
      self.actions = []

      if not self.needs_actions:
        self.drop()

  def get_state(self):
    return {"board": numpy.copy(self.board), 
            "tetromino": numpy.copy(self.tetromino),
            "tetromino_x": self.tetromino_x,
            "tetromino_y": self.tetromino_y,
            "score": self.score,
            "gameover": self.gameover,
            "needs_actions": self.needs_actions}

  def add_actions(self, new_actions):
    self.needs_actions = False
    self.actions = new_actions
    #print(self.actions)

if __name__ == '__main__':
  App = TetrisApp()
  App.run()

