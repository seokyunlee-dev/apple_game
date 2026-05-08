
import json
import logging
from dataclasses import dataclass
from typing import List, Tuple, Set

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class Solution:
    """합이 10이 되는 드래그 영역 정보를 담는 클래스"""
    r1: int  # 시작 행
    c1: int  # 시작 열
    r2: int  # 끝 행
    c2: int  # 끝 열
    sum: int # 합 (항상 10)
    apple_count: int  # 영역 내 0이 아닌 사과의 개수
    area: int # 영역의 크기 (가로 x 세로)

    def to_dict(self):
        return {
            "start": (self.r1, self.c1),
            "end": (self.r2, self.c2),
            "sum": self.sum,
            "apple_count": self.apple_count,
            "area": self.area
        }

def find_all_10_rectangles(grid: List[List[int]]) -> List[Solution]:
    """
    2차원 그리드에서 합이 10이 되는 모든 직사각형 영역을 찾습니다.
    
    Args:
        grid: 1~9 사이의 숫자로 구성된 2차원 리스트 (0은 빈 칸)
        
    Returns:
        합이 10인 모든 Solution 객체 리스트
    """
    if not grid or not grid[0]:
        return []

    rows = len(grid)
    cols = len(grid[0])
    solutions = []

    # 1. 효율적인 합 계산을 위한 2D Prefix Sum 구축
    prefix_sum = [[0] * (cols + 1) for _ in range(rows + 1)]
    # 2. 사과 개수 계산을 위한 Prefix Sum
    apple_count_prefix = [[0] * (cols + 1) for _ in range(rows + 1)]

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            prefix_sum[r + 1][c + 1] = val + prefix_sum[r][c + 1] + prefix_sum[r + 1][c] - prefix_sum[r][c]
            apple_count_prefix[r + 1][c + 1] = (1 if val > 0 else 0) + apple_count_prefix[r][c + 1] + apple_count_prefix[r + 1][c] - apple_count_prefix[r][c]

    def get_sum(r1, c1, r2, c2):
        """(r1, c1)에서 (r2, c2)까지의 직사각형 영역 합 반환"""
        return prefix_sum[r2 + 1][c2 + 1] - prefix_sum[r1][c2 + 1] - prefix_sum[r2 + 1][c1] + prefix_sum[r1][c1]

    def get_apple_count(r1, c1, r2, c2):
        """영역 내의 실제 사과 개수 반환"""
        return apple_count_prefix[r2 + 1][c2 + 1] - apple_count_prefix[r1][c2 + 1] - apple_count_prefix[r2 + 1][c1] + apple_count_prefix[r1][c1]

    # 3. 모든 가능한 직사각형 영역 탐색 (Brute Force with Optimization)
    for r1 in range(rows):
        for c1 in range(cols):
            # 시작점이 빈 칸(0)이면 스킵 (선택 사항이나 효율적)
            if grid[r1][c1] == 0:
                continue
                
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    current_sum = get_sum(r1, c1, r2, c2)
                    
                    if current_sum == 10:
                        count = get_apple_count(r1, c1, r2, c2)
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        solutions.append(Solution(r1, c1, r2, c2, current_sum, count, area))
                    elif current_sum > 10:
                        # 숫자가 모두 양수(1~9)이므로, 합이 10을 넘으면 더 확장할 필요 없음
                        break
    
    return solutions

def solve_apple_game(grid: List[List[int]]) -> List[Solution]:
    """
    그리드에서 최적의 사과 제거 순서를 결정합니다.
    우선순위:
    1. 없앨 수 있는 사과의 개수가 많은 영역 (apple_count DESC)
    2. 드래그 거리가 짧은 영역 (area ASC)
    
    겹치는 영역은 Greedy하게 제거됩니다.
    """
    all_solutions = find_all_10_rectangles(grid)
    
    # 정렬 기준: 1. 사과 개수(많을수록), 2. 면적(작을수록)
    sorted_solutions = sorted(
        all_solutions, 
        key=lambda x: (-x.apple_count, x.area)
    )
    
    final_solutions = []
    used_cells: Set[Tuple[int, int]] = set()

    for sol in sorted_solutions:
        # 현재 영역이 이미 사용된 셀과 겹치는지 확인
        is_overlapping = False
        for r in range(sol.r1, sol.r2 + 1):
            for c in range(sol.c1, sol.c2 + 1):
                if (r, c) in used_cells:
                    is_overlapping = True
                    break
            if is_overlapping:
                break
        
        if not is_overlapping:
            final_solutions.append(sol)
            # 사용된 셀 표시
            for r in range(sol.r1, sol.r2 + 1):
                for c in range(sol.c1, sol.c2 + 1):
                    used_cells.add((r, c))
    
    return final_solutions

def main():
    try:
        with open("grid.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            grid = data.get("grid", [])
    except Exception as e:
        logging.error(f"파일을 읽는 중 오류 발생: {e}")
        return

    if not grid:
        logging.warning("그리드 데이터를 찾을 수 없습니다.")
        return

    results = solve_apple_game(grid)
    
    logging.info(f"Optimal combinations found: {len(results)}")
    print("-" * 60)
    print(f"{'No':<4} | {'Start(r,c)':<12} | {'End(r,c)':<12} | {'Apples':<6} | {'Area':<4}")
    print("-" * 60)
    for i, sol in enumerate(results):
        start_str = f"({sol.r1},{sol.c1})"
        end_str = f"({sol.r2},{sol.c2})"
        print(f"{i+1:<4} | {start_str:<12} | {end_str:<12} | {sol.apple_count:>6} | {sol.area:>4}")
    print("-" * 60)

if __name__ == "__main__":
    main()
