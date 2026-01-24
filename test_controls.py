#!/usr/bin/env python3
"""
Snake Game AI 미로 기능 테스트
M키로 미로 토글, +/-키로 속도 조절 테스트
"""

import pygame
import sys
from snake_game import SnakeGameAI

def test_controls():
    """미로 토글과 속도 조절 기능 테스트"""
    print("Snake Game AI 미로 기능 테스트")
    print("컨트롤:")
    print("- M키: 미로 ON/OFF 토글")
    print("- +키: 속도 증가")
    print("- -키: 속도 감소")
    print("- ESC키: 종료")
    print("- 화면을 닫으면 종료")
    print()
    
    # 게임 초기화 (미로 모드 ON으로 시작)
    game = SnakeGameAI(enable_maze=True, speed=40)
    
    print(f"초기 설정: 미로={'ON' if game.enable_maze else 'OFF'}, 속도={game.speed}")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    game.toggle_maze()
                    print(f"미로 토글: {'ON' if game.enable_maze else 'OFF'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    game.increase_speed()
                    print(f"속도 증가: {game.speed}")
                elif event.key == pygame.K_MINUS:
                    game.decrease_speed()
                    print(f"속도 감소: {game.speed}")
        
        # 간단한 AI 동작 (직진)
        action = [1, 0, 0]  # 직진
        reward, done, score = game.play_step(action)
        
        if done:
            print(f"게임 종료 - 점수: {score}")
            game.reset()
        
        clock.tick(60)  # 60 FPS로 실행
    
    pygame.quit()
    print("테스트 완료!")

if __name__ == "__main__":
    test_controls()