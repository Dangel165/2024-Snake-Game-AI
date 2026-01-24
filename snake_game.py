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
    
    def __init__(self, w=640, h=480, enable_maze=True, speed=40):
        self.w = w
        self.h = h
        self.enable_maze = enable_maze
        self.speed = speed
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI By:Dangel')
        self.clock = pygame.time.Clock()
        self.obstacles = self._create_maze() if self.enable_maze else []
        self.reset()
        
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
        """미로 생성 - 내부 장애물만 생성"""
        obstacles = []
        
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
        
        # 랜덤 장애물 추가
        for _ in range(10):
            x = random.randint(2, (self.w // BLOCK_SIZE) - 3) * BLOCK_SIZE
            y = random.randint(2, (self.h // BLOCK_SIZE) - 3) * BLOCK_SIZE
            obstacle = Point(x, y)
            
            # 중앙 근처와 시작 위치 근처는 피하기
            if (abs(x - center_x) > BLOCK_SIZE * 2 or abs(y - center_y) > BLOCK_SIZE * 2) and \
               (abs(x - self.w//2) > BLOCK_SIZE * 4 or abs(y - self.h//2) > BLOCK_SIZE * 4):
                obstacles.append(obstacle)
        
        return obstacles
        
    def reset(self):
        self.direction = Direction.RIGHT
        
        # 시작 위치를 장애물과 겹치지 않게 설정
        start_x = BLOCK_SIZE * 4
        start_y = BLOCK_SIZE * 4
        
        self.head = Point(start_x, start_y)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        """음식을 장애물과 뱀이 없는 곳에 배치"""
        while True:
            # 경계에서 떨어진 곳에 배치 (항상)
            x = random.randint(1, (self.w-2*BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(1, (self.h-2*BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            food_point = Point(x, y)
            
            # 뱀과 겹치지 않는 위치 찾기
            if food_point not in self.snake:
                # 미로 모드일 때만 장애물 체크
                if not self.enable_maze or food_point not in self.obstacles:
                    self.food = food_point
                    break
        
    def play_step(self, action):
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:  # M키로 미로 토글
                    self.toggle_maze()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:  # +키로 속도 증가
                    self.increase_speed()
                elif event.key == pygame.K_MINUS:  # -키로 속도 감소
                    self.decrease_speed()
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(self.speed)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            
        # 경계 충돌 체크 (항상 적용)
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # 자기 자신과 충돌 체크
        if pt in self.snake[1:]:
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
        
        # 뱀 그리기
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # 음식 그리기
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 점수 표시
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        # 미로 상태 표시
        maze_status = "Maze: ON" if self.enable_maze else "Maze: OFF"
        maze_text = font.render(maze_status, True, GREEN if self.enable_maze else RED)
        self.display.blit(maze_text, [self.w - 120, 0])
        
        # 속도 표시
        speed_text = font.render(f"Speed: {self.speed}", True, WHITE)
        self.display.blit(speed_text, [self.w - 120, 25])
        
        # 컨트롤 안내
        control_text = font.render("M: Toggle Maze, +/-: Speed", True, WHITE)
        self.display.blit(control_text, [10, self.h - 25])
        
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
