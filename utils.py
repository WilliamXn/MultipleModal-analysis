import numpy as np

def getPairs(points):
    paris = []
    for ind in range(0, 21):
        paris.append([points[ind * 2], points[ind * 2 + 1]])
    return paris, points[-1]


def centralize(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return [[x - x_mean, y - y_mean] for x, y in points]


def rotate(points):
    start_vec = points[0]
    end_vec = points[13]
    vec = [end_vec[0] - start_vec[0], end_vec[1] - start_vec[1]]
    vec = [end_vec[0] - start_vec[0], end_vec[1] - start_vec[1]]
    angle = np.arctan2(vec[1], vec[0])
    return [[x * np.cos(angle) + y * np.sin(angle), -x * np.sin(angle) + y * np.cos(angle)] for x, y in points]


def vec_length(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def scale(points, scl=0.01):
    start_vec = points[0]
    end_vec = points[13]
    vec = [end_vec[0] - start_vec[0], end_vec[1] - start_vec[1]]
    length = np.sqrt(vec[0] ** 2 + vec[1] ** 2) * scl
    return [
        [x * vec_length(x, y) / length, y * vec_length(x, y) / length]
        for x, y in points
    ]


def decompose(points):
    flatten = []
    for point in points:
        flatten.append(point[0])
        flatten.append(point[1])
    return flatten


def process_line(points, scl=0.1):
    return decompose(scale(rotate(centralize(points)), scl=scl))