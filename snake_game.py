import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    
    def __init__(self, w=640, h=480, enable_maze=False, speed=40, difficulty='easy', num_players=1):
        self.w = w
        self.h = h
        self.enable_maze = enable_maze  # 기본값을 False로 변경
        self.speed = speed
        self.difficulty = difficulty  # 'easy', 'medium', 'hard'
        self.num_players = num_players
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI By:Dangel')
        self.clock = pygame.time.Clock()
        self.obstacles = self._create_maze() if self.enable_maze else []
        
        # 멀티플레이어를 위한 뱀들 초기화
        self.snakes = []
        self.scores = []
        self.foods = []
        self.frame_iterations = []
        
        self.reset()
        
    def set_difficulty(self, difficulty):
        """난이도 설정"""
        self.difficulty = difficulty
        if difficulty == 'easy':
            self.speed = max(20, self.speed)
        elif difficulty == 'medium':
            self.speed = max(40, self.speed)
        elif difficulty == 'hard':
            self.speed = max(60, self.speed)
        
        # 난이도에 따른 장애물 수 조정
        if self.enable_maze:
            self.obstacles = self._create_maze()
            
    def cycle_difficulty(self):
        """난이도 순환 (쉬움 → 보통 → 어려움 → 쉬움)"""
        difficulties = ['easy', 'medium', 'hard']
        current_idx = difficulties.index(self.difficulty)
        next_idx = (current_idx + 1) % len(difficulties)
        self.set_difficulty(difficulties[next_idx])
        
    def toggle_maze(self):
        """미로 on/off 토글"""
        self.enable_maze = not self.enable_maze
        self.obstacles = self._create_maze() if self.enable_maze else []
        # 현재 뱀이 장애물과 겹치면 리셋
        if self.enable_maze and any(pt in self.obstacles for pt in self.snake):
            self.reset()
        # 음식이 장애물과 겹치면 재배치
        if self.enable_maze and self.food in self.obstacles:
            self._place_food()
            
    def toggle_multiplayer(self):
        """멀티플레이어 on/off 토글"""
        if self.num_players == 1:
            self.num_players = 2  # 싱글 → 멀티
        else:
            self.num_players = 1  # 멀티 → 싱글
        
        # 게임 리셋으로 새로운 플레이어 수 적용
        self.reset()
        print(f"멀티플레이어 모드: {'ON' if self.num_players > 1 else 'OFF'} ({self.num_players}명)")
            
    def set_speed(self, speed):
        """속도 설정 (10-100)"""
        self.speed = max(10, min(100, speed))
        
    def increase_speed(self):
        """속도 증가"""
        self.speed = min(100, self.speed + 10)
        
    def decrease_speed(self):
        """속도 감소"""
        self.speed = max(10, self.speed - 10)
        
    def _create_maze(self):
        """미로 생성 - 난이도에 따른 장애물 수 조정"""
        obstacles = []
        
        # 난이도별 장애물 수
        obstacle_counts = {
            'easy': 5,
            'medium': 10,
            'hard': 15
        }
        
        # 내부 장애물 패턴
        # 십자 모양 장애물
        center_x = self.w // 2
        center_y = self.h // 2
        
        # 수직 장애물
        for i in range(-3, 4):
            if center_y + i * BLOCK_SIZE > BLOCK_SIZE and center_y + i * BLOCK_SIZE < self.h - 2 * BLOCK_SIZE:
                obstacles.append(Point(center_x, center_y + i * BLOCK_SIZE))
        
        # 수평 장애물
        for i in range(-3, 4):
            if center_x + i * BLOCK_SIZE > BLOCK_SIZE and center_x + i * BLOCK_SIZE < self.w - 2 * BLOCK_SIZE:
                obstacles.append(Point(center_x + i * BLOCK_SIZE, center_y))
        
        # 모서리 L자 장애물
        # 좌상단
        for i in range(3):
            obstacles.append(Point(BLOCK_SIZE * 3, BLOCK_SIZE * (2 + i)))
            obstacles.append(Point(BLOCK_SIZE * (3 + i), BLOCK_SIZE * 2))
        
        # 우상단
        for i in range(3):
            obstacles.append(Point(self.w - BLOCK_SIZE * 4, BLOCK_SIZE * (2 + i)))
            obstacles.append(Point(self.w - BLOCK_SIZE * (4 + i), BLOCK_SIZE * 2))
        
        # 좌하단
        for i in range(3):
            obstacles.append(Point(BLOCK_SIZE * 3, self.h - BLOCK_SIZE * (3 + i)))
            obstacles.append(Point(BLOCK_SIZE * (3 + i), self.h - BLOCK_SIZE * 3))
        
        # 우하단
        for i in range(3):
            obstacles.append(Point(self.w - BLOCK_SIZE * 4, self.h - BLOCK_SIZE * (3 + i)))
            obstacles.append(Point(self.w - BLOCK_SIZE * (4 + i), self.h - BLOCK_SIZE * 3))
        
        # 난이도별 랜덤 장애물 추가
        random_count = obstacle_counts.get(self.difficulty, 10)
        for _ in range(random_count):
            x = random.randint(2, (self.w // BLOCK_SIZE) - 3) * BLOCK_SIZE
            y = random.randint(2, (self.h // BLOCK_SIZE) - 3) * BLOCK_SIZE
            obstacle = Point(x, y)
            
            # 중앙 근처와 시작 위치 근처는 피하기
            if (abs(x - center_x) > BLOCK_SIZE * 2 or abs(y - center_y) > BLOCK_SIZE * 2) and \
               (abs(x - self.w//2) > BLOCK_SIZE * 4 or abs(y - self.h//2) > BLOCK_SIZE * 4):
                obstacles.append(obstacle)
        
        return obstacles
        
    def reset(self):
        # 멀티플레이어를 위한 방향 초기화
        self.directions = []
        
        # 멀티플레이어를 위한 초기화
        self.snakes = []
        self.scores = []
        self.foods = []
        self.frame_iterations = []
        
        for i in range(self.num_players):
            # 각 플레이어별 방향 설정
            self.directions.append(Direction.RIGHT)
            
            # 시작 위치를 플레이어별로 다르게 설정
            start_x = BLOCK_SIZE * (4 + i * 6)
            start_y = BLOCK_SIZE * (4 + i * 2)
            
            # 화면 범위를 벗어나지 않도록 조정
            if start_x >= self.w - BLOCK_SIZE * 3:
                start_x = BLOCK_SIZE * 4
                start_y += BLOCK_SIZE * 4
            
            head = Point(start_x, start_y)
            snake = [head,
                    Point(head.x-BLOCK_SIZE, head.y),
                    Point(head.x-(2*BLOCK_SIZE), head.y)]
            
            self.snakes.append(snake)
            self.scores.append(0)
            self.frame_iterations.append(0)
        
        # 첫 번째 플레이어를 메인으로 설정 
        if self.snakes:
            self.direction = self.directions[0] if self.directions else Direction.RIGHT
            self.snake = self.snakes[0]
            self.head = self.snake[0]
            self.score = self.scores[0]
            self.frame_iteration = self.frame_iterations[0]
        
        # 음식 배치
        self.foods = []
        for _ in range(self.num_players):
            self._place_food()
        
    def _place_food(self, player_idx=None):
        """음식을 장애물과 뱀이 없는 곳에 배치"""
        while True:
            # 경계에서 떨어진 곳에 배치 
            x = random.randint(1, (self.w-2*BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(1, (self.h-2*BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            food_point = Point(x, y)
            
            # 모든 뱀과 겹치지 않는 위치 찾기
            collision = False
            for snake in self.snakes:
                if food_point in snake:
                    collision = True
                    break
            
            if not collision:
                # 미로 모드일 때만 장애물 체크
                if not self.enable_maze or food_point not in self.obstacles:
                    if player_idx is not None and player_idx < len(self.foods):
                        self.foods[player_idx] = food_point
                    else:
                        if len(self.foods) < self.num_players:
                            self.foods.append(food_point)
                        else:
                            self.foods[0] = food_point
                    
                    # 기존 호환성을 위해 첫 번째 음식을 메인으로 설정
                    if self.foods:
                        self.food = self.foods[0]
                    break
        
    def play_step(self, action, player_id=0):
        # 플레이어 ID 유효성 검사
        if player_id >= len(self.snakes) or player_id >= len(self.frame_iterations) or player_id >= len(self.scores):
            # 유효하지 않은 player_id인 경우 기본값 반환
            return 0, True, 0
        
        # 지정된 플레이어의 frame_iteration 증가
        if player_id < len(self.frame_iterations):
            self.frame_iterations[player_id] += 1
        
        # 메인 플레이어의 frame_iteration도 업데이트 
        self.frame_iteration = self.frame_iterations[0] if self.frame_iterations else 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:  # M키로 미로 토글
                    self.toggle_maze()
                elif event.key == pygame.K_p:  # P키로 멀티플레이어 토글
                    self.toggle_multiplayer()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:  # +키로 속도 증가
                    self.increase_speed()
                elif event.key == pygame.K_MINUS:  # -키로 속도 감소
                    self.decrease_speed()
                elif event.key == pygame.K_d:  # D키로 난이도 변경
                    self.cycle_difficulty()
        
        # 지정된 플레이어 이동
        self._move(action, player_id)
        
        # 해당 플레이어의 뱀 업데이트
        if player_id < len(self.snakes):
            current_snake = self.snakes[player_id]
            current_head = current_snake[0]  # _move에서 이미 업데이트된 새로운 머리 위치
            
            # 메인 플레이어 정보 업데이트 
            if player_id == 0:
                self.snake = current_snake
                self.head = current_head
        
        reward = 0
        game_over = False
        
        # 충돌 검사 
        collision_check = False
        timeout_check = False
        
        if player_id < len(self.frame_iterations) and player_id < len(self.snakes):
            collision_check = self.is_collision(player_id=player_id)
            timeout_check = self.frame_iterations[player_id] > 100 * len(self.snakes[player_id])
        
        if collision_check or timeout_check:
            game_over = True
            reward = -10
            score = self.scores[player_id] if player_id < len(self.scores) else 0
            return reward, game_over, score
        
        # 음식 먹기 검사 
        if player_id < len(self.snakes) and player_id < len(self.scores):
            current_head = self.snakes[player_id][0]
            if player_id < len(self.foods) and current_head == self.foods[player_id]:
                self.scores[player_id] += 1
                reward = 10
                self._place_food(player_id)
                
                # 메인 플레이어 점수 업데이트 
                if player_id == 0:
                    self.score = self.scores[0]
            else:
                # 꼬리 제거
                self.snakes[player_id].pop()
        
        self._update_ui()
        self.clock.tick(self.speed)
        
        # 안전한 점수 반환
        final_score = self.scores[player_id] if player_id < len(self.scores) else 0
        return reward, game_over, final_score
    
    def is_collision(self, pt=None, player_id=0):
        if pt is None:
            if player_id < len(self.snakes):
                pt = self.snakes[player_id][0]
            else:
                pt = self.head
            
        # 경계 충돌 체크 
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # 자기 자신과 충돌 체크
        if player_id < len(self.snakes):
            current_snake = self.snakes[player_id]
            if pt in current_snake[1:]:
                return True
        
        # 멀티플레이어에서 다른 플레이어와 충돌 체크
        if self.num_players > 1:
            for other_player_id, other_snake in enumerate(self.snakes):
                if other_player_id != player_id:  # 자기 자신 제외
                    if pt in other_snake:  # 다른 플레이어의 몸체와 충돌
                        return True
        
        # 미로 모드일 때만 장애물과 충돌 체크
        if self.enable_maze and pt in self.obstacles:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # 미로 모드일 때만 장애물 그리기
        if self.enable_maze:
            for obstacle in self.obstacles:
                pygame.draw.rect(self.display, GRAY, pygame.Rect(obstacle.x, obstacle.y, BLOCK_SIZE, BLOCK_SIZE))
                # 장애물에 테두리 추가
                pygame.draw.rect(self.display, WHITE, pygame.Rect(obstacle.x, obstacle.y, BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # 모든 뱀 그리기 (멀티플레이어)
        colors = [(BLUE1, BLUE2), (GREEN, WHITE), (RED, WHITE), (WHITE, BLACK)]
        for i, snake in enumerate(self.snakes):
            color1, color2 = colors[i % len(colors)]
            for pt in snake:
                pygame.draw.rect(self.display, color1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, color2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # 모든 음식 그리기
        for i, food in enumerate(self.foods):
            food_color = RED if i == 0 else (255, 165, 0)  # 첫 번째는 빨강, 나머지는 주황
            pygame.draw.rect(self.display, food_color, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 점수 표시
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        # 멀티플레이어 점수 표시
        if self.num_players > 1:
            for i, score in enumerate(self.scores):
                score_text = font.render(f"P{i+1}: {score}", True, WHITE)
                self.display.blit(score_text, [0, 25 + i * 25])
        
        # 미로 상태 표시
        maze_status = "Maze: ON" if self.enable_maze else "Maze: OFF"
        maze_text = font.render(maze_status, True, GREEN if self.enable_maze else RED)
        self.display.blit(maze_text, [self.w - 120, 0])
        
        # 속도 표시
        speed_text = font.render(f"Speed: {self.speed}", True, WHITE)
        self.display.blit(speed_text, [self.w - 120, 25])
        
        # 난이도 표시
        difficulty_text = font.render(f"Difficulty: {self.difficulty.title()}", True, WHITE)
        self.display.blit(difficulty_text, [self.w - 150, 50])
        
        # 플레이어 수 표시
        if self.num_players > 1:
            players_text = font.render(f"Players: {self.num_players}", True, WHITE)
            self.display.blit(players_text, [self.w - 120, 75])
        
        # 컨트롤 안내
        control_text = font.render("M: Maze, P: Multiplayer, +/-: Speed, D: Difficulty", True, WHITE)
        self.display.blit(control_text, [10, self.h - 25])
        
        pygame.display.flip()
        
    def _move(self, action, player_id=0):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        # 현재 플레이어의 방향 가져오기
        if player_id < len(self.directions):
            current_direction = self.directions[player_id]
        else:
            current_direction = Direction.RIGHT
            
        idx = clock_wise.index(current_direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        # 플레이어별 방향 업데이트
        if player_id < len(self.directions):
            self.directions[player_id] = new_dir
        
        # 메인 플레이어 방향 업데이트 
        if player_id == 0:
            self.direction = new_dir
        
        # 현재 플레이어의 머리 위치 가져오기
        if player_id < len(self.snakes):
            current_head = self.snakes[player_id][0]
            x = current_head.x
            y = current_head.y
        else:
            x = self.head.x
            y = self.head.y
            
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
        
        new_head = Point(x, y)
        
        # 플레이어별 새로운 머리 위치를 뱀의 앞쪽에 삽입
        if player_id < len(self.snakes):
            self.snakes[player_id].insert(0, new_head)
        
        # 메인 플레이어 머리 위치 업데이트 
        if player_id == 0:
            self.head = new_head
