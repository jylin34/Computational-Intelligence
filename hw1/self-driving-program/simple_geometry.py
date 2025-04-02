import math as m


class Point2D:
    ...


class Line2D:
    ...


class Point2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def length(self):  # length when seem as a vector
        return m.sqrt(self.x**2 + self.y**2)

    def __str__(self) -> str:
        return f'{self.x} {self.y}'

    def __sub__(self, point):
        difx = self.x - point.x
        dify = self.y - point.y
        return Point2D(difx, dify)

    def __add__(self, point):
        sumx = self.x + point.x
        sumy = self.y + point.y
        return Point2D(sumx, sumy)

    def __mul__(self, num: float):
        return Point2D(self.x*num, self.y*num)

    def __div__(self, num: float):
        return Point2D(self.x/num, self.y/num)

    def distToPoint2D(self, p2: Point2D):
        diff = self - p2
        return diff.length

    def distToLine2D(self, line: Line2D):
        p1 = line.p1
        lineTop1 = Line2D(self, p1)
        angle = lineTop1.angleToLine(line)
        return m.sin(angle/180*m.pi) * lineTop1.length

    def rorate(self, angle) -> Point2D:
        '''
        |cos a -sin a|       |x|
        |sin a  cos a| = RM, |y| = P
        |newx|
        |newy| = RM @ P

        new_x = x*cos(a) - y*sin(a)
        new_y = x*sin(a) + y*cos(a)
        '''
        rad = angle/180*m.pi
        new_x = self.x*m.cos(rad) - self.y*m.sin(rad)
        new_y = self.x*m.sin(rad) - self.y*m.cos(rad)
        return Point2D(new_x, new_y)

    def isInRect(self, p1, p2):
        if p1.x > p2.x:
            rx = p1.x
            lx = p2.x
        else:
            rx = p2.x
            lx = p1.x

        if p1.y > p2.y:
            uy = p1.y
            dy = p2.y
        else:
            uy = p2.y
            dy = p1.y

        return lx <= self.x <= rx and dy <= self.y <= uy


class Line2D():
    def __init__(self, *arg):
        if len(arg) == 2:
            self.p1 = arg[0]
            self.p2 = arg[1]
        else:
            self.p1 = Point2D(arg[0], arg[1])
            self.p2 = Point2D(arg[2], arg[3])

    @property
    def length(self):
        return (self.p1-self.p2).length

    def __str__(self) -> str:
        return f'{self.p1} {self.p2}'

    # find angle be between two lines
    def angleToLine(self, line: Line2D):
        p1, p2 = line.p1, line.p2
        p3, p4 = self.p1, self.p2

        v1 = p1-p2
        len_line = v1.length

        v2 = p3-p4
        len_line2 = v2.length
        angle_diff = m.acos((v1.x * v2.x + v1.y * v2.y) /
                            (len_line * len_line2 + 1e-10))

        return angle_diff/m.pi*180

    def lineOverlap(self, line2: Line2D):
        '''
        input:
            line2: Line2D
                Check if this line is overlapped with line2
        output:
            isOverlapped: bool
                if two line are overlapped, return True

            t: float
                [x1 + t(x2-x1), y1 + t(y2-y1)]
            u: float
                [x3 + u(x4-x3), y3 + u(y4-y3)]
        '''

        p1, p2 = self.p1, self.p2
        p3, p4 = line2.p1, line2.p2

        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y

        # point intersect = [x1 + t(x2-x1), y1 + t(y2-y1)] = [x3 + u(x4-x3), y3 + u(y4-y3)]
        x13 = x1 - x3
        x21 = x2 - x1
        x43 = x4 - x3
        y21 = y2 - y1
        x34 = x3 - x4
        y34 = y3 - y4
        y31 = y3 - y1

        try:
            t = (x13*y34 + y31*x34)/(-x21*y34+y21*x34)
            u = (x13*y21 + y31*x21)/(x43*y21+y34*x21)
        except ZeroDivisionError as e:
            if (x13*y34 + y31*x34) == 0 and (x13*y21 + y31*x21) == 0:
                return True, None, None  # two lines are parallel and overlapped
            else:
                # two lines are parallel but no overlapped, or they intersect at the point out of range
                return False, None, None

        if 0 <= t <= 1 and 0 <= u <= 1:
            return True, t, u
        else:
            return False, t, u
