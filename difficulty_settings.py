#!/usr/bin/env python3
"""
Snake Game AI 난이도 설정
"""

# 난이도별 설정
DIFFICULTY_SETTINGS = {
    'easy': {
        'min_speed': 20,
        'default_speed': 30,
        'obstacle_count': 5,
        'description': '초보자용 - 적은 장애물, 느린 속도',
        'color': (0, 255, 0)  # 초록색
    },
    'medium': {
        'min_speed': 40,
        'default_speed': 50,
        'obstacle_count': 10,
        'description': '중급자용 - 보통 장애물, 중간 속도',
        'color': (255, 255, 0)  # 노란색
    },
    'hard': {
        'min_speed': 60,
        'default_speed': 70,
        'obstacle_count': 15,
        'description': '고급자용 - 많은 장애물, 빠른 속도',
        'color': (255, 0, 0)  # 빨간색
    },
    'extreme': {
        'min_speed': 80,
        'default_speed': 90,
        'obstacle_count': 20,
        'description': '전문가용 - 매우 많은 장애물, 매우 빠른 속도',
        'color': (128, 0, 128)  # 보라색
    }
}

def get_difficulty_info(difficulty):
    """난이도 정보 반환"""
    return DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS['easy'])

def get_all_difficulties():
    """모든 난이도 목록 반환"""
    return list(DIFFICULTY_SETTINGS.keys())

def get_next_difficulty(current_difficulty):
    """다음 난이도 반환 (순환)"""
    difficulties = get_all_difficulties()
    try:
        current_idx = difficulties.index(current_difficulty)
        next_idx = (current_idx + 1) % len(difficulties)
        return difficulties[next_idx]
    except ValueError:
        return 'easy'

if __name__ == "__main__":
    print("Snake Game AI 난이도 설정")
    print("=" * 40)
    
    for difficulty, settings in DIFFICULTY_SETTINGS.items():
        print(f"\n{difficulty.upper()}:")
        print(f"  설명: {settings['description']}")
        print(f"  기본 속도: {settings['default_speed']}")
        print(f"  최소 속도: {settings['min_speed']}")
        print(f"  장애물 수: {settings['obstacle_count']}")