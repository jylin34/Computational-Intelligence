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
