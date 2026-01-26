import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(27, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def get_state(self, game, player_id=0):
        # 지정된 플레이어의 뱀 정보 가져오기
        if player_id < len(game.snakes):
            head = game.snakes[player_id][0]
            current_direction = game.directions[player_id] if player_id < len(game.directions) else Direction.RIGHT
        else:
            head = game.snake[0]
            current_direction = game.direction
            
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = current_direction == Direction.LEFT
        dir_r = current_direction == Direction.RIGHT
        dir_u = current_direction == Direction.UP
        dir_d = current_direction == Direction.DOWN
        
        # 기본 충돌 감지 
        state = [
            (dir_r and game.is_collision(point_r, player_id)) or 
            (dir_l and game.is_collision(point_l, player_id)) or 
            (dir_u and game.is_collision(point_u, player_id)) or 
            (dir_d and game.is_collision(point_d, player_id)),
            
            (dir_u and game.is_collision(point_r, player_id)) or 
            (dir_d and game.is_collision(point_l, player_id)) or 
            (dir_l and game.is_collision(point_u, player_id)) or 
            (dir_r and game.is_collision(point_d, player_id)),
            
            (dir_d and game.is_collision(point_r, player_id)) or 
            (dir_u and game.is_collision(point_l, player_id)) or 
            (dir_r and game.is_collision(point_u, player_id)) or 
            (dir_l and game.is_collision(point_d, player_id)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]
        
        # 음식 위치 (해당 플레이어의 음식)
        if player_id < len(game.foods):
            food = game.foods[player_id]
        else:
            food = game.food
            
        state.extend([
            food.x < head.x,
            food.x > head.x,
            food.y < head.y,
            food.y > head.y
        ])
        
        # 장애물 감지 추가 (미로 모드일 때만)
        if game.enable_maze:
            obstacle_detection = [
                # 직선 방향 장애물 감지
                point_l in game.obstacles,  # 왼쪽
                point_r in game.obstacles,  # 오른쪽
                point_u in game.obstacles,  # 위쪽
                point_d in game.obstacles,  # 아래쪽
                
                # 대각선 방향 장애물 감지
                Point(head.x - 20, head.y - 20) in game.obstacles,  # 좌상
                Point(head.x + 20, head.y - 20) in game.obstacles,  # 우상
                Point(head.x - 20, head.y + 20) in game.obstacles,  # 좌하
                Point(head.x + 20, head.y + 20) in game.obstacles,  # 우하
            ]
        else:
            # 미로 모드가 아닐 때는 모든 장애물 감지를 False로
            obstacle_detection = [False] * 8
        
        # 상태 결합
        state.extend(obstacle_detection)
        
        # 멀티플레이어에서 다른 플레이어 감지 추가
        if game.num_players > 1:
            other_player_detection = []
            
            # 주변에 다른 플레이어가 있는지 감지 
            directions_8 = [
                Point(head.x - 20, head.y),      # 왼쪽
                Point(head.x + 20, head.y),      # 오른쪽
                Point(head.x, head.y - 20),      # 위쪽
                Point(head.x, head.y + 20),      # 아래쪽
                Point(head.x - 20, head.y - 20), # 좌상
                Point(head.x + 20, head.y - 20), # 우상
                Point(head.x - 20, head.y + 20), # 좌하
                Point(head.x + 20, head.y + 20), # 우하
            ]
            
            for check_point in directions_8:
                other_player_nearby = False
                for other_player_id, other_snake in enumerate(game.snakes):
                    if other_player_id != player_id and check_point in other_snake:
                        other_player_nearby = True
                        break
                other_player_detection.append(other_player_nearby)
            
            state.extend(other_player_detection)
        else:
            # 싱글플레이어에서는 다른 플레이어 감지를 모두 False로
            state.extend([False] * 8)
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    # 최대 4명의 AI 에이전트 생성
    agents = [Agent() for _ in range(4)]
    
    game = SnakeGameAI(enable_maze=False, speed=40, difficulty='easy')
    
    print("Snake AI Training Started!")
    print("Controls during training:")
    print("- M: Toggle Maze ON/OFF")
    print("- P: Toggle Multiplayer ON/OFF")
    print("- +: Increase Speed")
    print("- -: Decrease Speed")
    print("- D: Change Difficulty (Easy/Medium/Hard/Extreme)")
    print("- Close window to stop training")
    
    while True:
        if game.num_players == 1:
            # 싱글플레이어 모드 
            state_old = agents[0].get_state(game, player_id=0)
            final_move = agents[0].get_action(state_old)
            reward, done, score = game.play_step(final_move, player_id=0)
            state_new = agents[0].get_state(game, player_id=0)
            
            agents[0].train_short_memory(state_old, final_move, reward, state_new, done)
            agents[0].remember(state_old, final_move, reward, state_new, done)
            
            # 게임 종료 처리
            if done:
                game.reset()
                agents[0].n_games += 1
                agents[0].train_long_memory()
                
                if score > record:
                    record = score
                    try:
                        agents[0].model.save()
                        print(f"New record! Model saved: {record}")
                    except Exception as e:
                        print(f"Failed to save model: {e}")
                    
                print('Game', agents[0].n_games, 'Score', score, 'Record:', record, 
                      f'Maze: {"ON" if game.enable_maze else "OFF"}', 
                      f'Speed: {game.speed}', f'Difficulty: {game.difficulty.title()}',
                      f'Players: {game.num_players}')
                
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agents[0].n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
        
        else:
            # 멀티플레이어 모드 
            states_old = []
            actions = []
            
            # 각 플레이어의 상태 수집 및 행동 결정
            for player_id in range(game.num_players):
                if player_id < len(game.snakes) and player_id < len(agents):
                    state_old = agents[player_id].get_state(game, player_id=player_id)
                    states_old.append(state_old)
                    
                    final_move = agents[player_id].get_action(state_old)
                    actions.append(final_move)
            
            # 모든 플레이어 동시 행동 실행
            rewards = []
            dones = []
            scores = []
            
            for player_id in range(len(actions)):
                if player_id < len(game.snakes):
                    reward, done, score = game.play_step(actions[player_id], player_id)
                    rewards.append(reward)
                    dones.append(done)
                    scores.append(score)
            
            # 각 플레이어의 새로운 상태 수집
            states_new = []
            for player_id in range(len(states_old)):
                if player_id < len(game.snakes):
                    state_new = agents[player_id].get_state(game, player_id=player_id)
                    states_new.append(state_new)
            
            # 각 AI 에이전트 학습
            for player_id in range(min(len(states_old), len(states_new), len(actions), len(rewards), len(dones))):
                if player_id < len(agents):
                    agents[player_id].train_short_memory(states_old[player_id], actions[player_id], 
                                                       rewards[player_id], states_new[player_id], dones[player_id])
                    agents[player_id].remember(states_old[player_id], actions[player_id], 
                                             rewards[player_id], states_new[player_id], dones[player_id])
            
            # 게임 종료 처리 
            if any(dones):
                game.reset()
                
                # 모든 에이전트의 게임 수 증가 및 장기 메모리 학습
                for player_id in range(min(game.num_players, len(agents))):
                    agents[player_id].n_games += 1
                    agents[player_id].train_long_memory()
                
                # 최고 점수 기록 (모든 플레이어 중 최고점)
                max_score = max(scores) if scores else 0
                if max_score > record:
                    record = max_score
                    # 최고 점수를 낸 플레이어의 모델 저장
                    best_player = scores.index(max_score)
                    try:
                        agents[best_player].model.save(f'model_player_{best_player+1}.pth')
                        print(f"New record by Player {best_player+1}! Model saved: {record}")
                    except Exception as e:
                        print(f"Failed to save model: {e}")
                    
                # 게임 정보 출력
                game_num = agents[0].n_games
                score_info = ', '.join([f'P{i+1}:{scores[i]}' for i in range(len(scores))])
                print(f'Game {game_num} - Scores: [{score_info}] Record: {record}')
                print(f'Settings: Maze: {"ON" if game.enable_maze else "OFF"}, '
                      f'Speed: {game.speed}, Difficulty: {game.difficulty.title()}, '
                      f'Players: {game.num_players}')
                
                # 메인 플레이어(첫 번째)의 점수로 그래프 업데이트
                main_score = scores[0] if scores else 0
                plot_scores.append(main_score)
                total_score += main_score
                mean_score = total_score / game_num
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
