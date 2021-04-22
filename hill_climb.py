# tetris game and AI
#from tetris import TetrisApp
from tetris_trainer import TetrisApp
from tetris_ai import TetrisAI
from multiprocessing import Process
# from dqn_agent import DQNAgent
from datetime import datetime
import time, threading
from logs import CustomTensorBoard
from tqdm import tqdm
from collections import defaultdict, OrderedDict


if __name__ == '__main__':
    current_state = ai.tetris_app.get_state()
    ai.set_board(current_state["board"])
    ai.set_stone(current_state["stone"], current_state["stone_x"], current_state["stone_y"])
    # if not current_state["needs_actions"]:
    #     continue
    actions = []

    if current_state["gameover"] and training:
        ai.load_next_unit( current_state["score"] )
        actions.append("space")


    print("here")
    next_states = ai.get_possible_boards()
    print("Number of possible boards", len(next_states))
    board_scores = ai.get_board_scores_hc(next_states)

   actions.extend()
   

    print("no")
    # board_scores = ai.get_board_scores(next_states)
    # print(board_scores)
    # best_state = agent.best_state(next_states)
    print("yes")
    
    #         best_action = None
    #         for action, state in next_states.items():
    #             if state == best_state:
    #                 best_action = action
    #                 break

    #         reward, done = env.play(best_action[0], best_action[1], render=render,
    #                                 render_delay=render_delay)
            
    #         agent.add_to_memory(current_state, next_states[best_action], reward, done)
    #         current_state = next_states[best_action]
    #         steps += 1

    #     scores.append(env.get_game_score())

    #     # Train
    #     if episode % train_every == 0:
    #         agent.train(batch_size=batch_size, epochs=epochs)
    #         # log._plot()


    #     # Save for plot
    #     if episode % 5 == 0:
    #         # Logs
    #         avg_score = mean(scores[-5:])
    #         min_score = min(scores[-5:])
    #         max_score = max(scores[-5:])
    #         log._add(episode, avg_score, max_score, min_score)
    #         log.log(episode, avg_score, max_score, min_score)

    #     # # Save model
    #     # if episode % 50 == 0:
    #     #     agent.model.save('trained_models/rb_dqa_model.h5')





