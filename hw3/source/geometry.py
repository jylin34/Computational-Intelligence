import math

def parse_track_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    start = tuple(map(float, lines[0].split(',')))
    start_tl = (-6.0, 0.0)
    start_br = (6.0, 0.0)
    goal_tl = tuple(map(float, lines[1].split(',')))
    goal_br = tuple(map(float, lines[2].split(',')))
    border = [tuple(map(float, line.split(','))) for line in lines[3:]]

    return start, start_tl, start_br, goal_tl, goal_br, border

def cast_ray(x, y, angle_deg, border_segments, max_distance=1000):
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    min_dist = max_distance

    for x1, y1, x2, y2 in border_segments:
        hit, dist = ray_segment_intersect(x, y, dx, dy, x1, y1, x2, y2)
        if hit and dist < min_dist:
            min_dist = dist

    return min_dist

def ray_segment_intersect(x, y, dx, dy, x1, y1, x2, y2):
    # 射線方向向量 (dx, dy)
    # 線段向量 (sx, sy)
    sx = x2 - x1
    sy = y2 - y1

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-8:
        return False, None  # 平行，不相交

    t = ((x1 - x) * sy - (y1 - y) * sx) / denom
    u = ((x1 - x) * dy - (y1 - y) * dx) / denom

    if t >= 0 and 0 <= u <= 1:
        # 相交，距離 = t * 向量長
        return True, t * math.hypot(dx, dy)
    else:
        return False, None

def point_near_segment(px, py, x1, y1, x2, y2, tolerance): # 如果回傳True 代表感測點已經靠近牆壁
    # 線段長度為 0（起點 = 終點）就直接跳過
    if x1 == x2 and y1 == y2:
        return False

    # 投影法求距離
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    if t < 0.0 or t > 1.0:
        return False

    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    dist_sq = (px - nearest_x)**2 + (py - nearest_y)**2
    return dist_sq <= tolerance ** 2

def border_to_segments(border_points): # 輸入所有邊界點, 然後輸出線段的list
    segments = []
    for i in range(len(border_points)):
        x1, y1 = border_points[i]
        x2, y2 = border_points[(i + 1) % len(border_points)]
        segments.append((x1, y1, x2, y2))
    return segments

def is_circle_near_segment(cx, cy, radius, x1, y1, x2, y2):
    # 計算最近點到線段的距離
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        # 線段是一個點
        dist_sq = (cx - x1) ** 2 + (cy - y1) ** 2
        return dist_sq <= radius ** 2

    t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / length_sq))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    dist_sq = (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2

    return dist_sq <= radius ** 2
