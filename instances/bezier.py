from PIL import Image
from PIL import ImageDraw
import numpy as np

def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n):
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result


def supersample(list_of_tuples, ratio):
    out_list = []
    for tp in list_of_tuples:
        out_list.append((tp[0]*ratio, tp[1]*ratio))
    return out_list


def angles_to_point(angles, square_size, end_pt_offset):
    center = float(square_size)/2 - 0.5
    radius = center - end_pt_offset
    points = []

    for angle in angles:
        x = int(center + radius*np.cos(angle*np.pi/180))
        y = int(center + radius*np.sin(angle*np.pi/180))
        points.append((x,y))

    return points


def sample_coord(square_size, end_pt_offset):
    center = float(square_size)/2 - 0.5
    threshold_rad = center - end_pt_offset

    while True:
        x = np.random.randint(low = end_pt_offset, high = square_size-end_pt_offset)
        y = np.random.randint(low = end_pt_offset, high = square_size-end_pt_offset)
        rad = np.linalg.norm([x-center, y-center])
        if rad > threshold_rad:
            break

    return (x,y)


def generate_item(square_size, num_endpoints, num_ctrl_points, end_pt_offset, control_pt_offset, num_lines, thickness, ss):
    im = Image.new('RGB', (square_size*ss, square_size*ss), 'white')
    draw = ImageDraw.Draw(im)

    ts = [t/float(num_lines) for t in range(num_lines+1)]

    while True:
        endpoints_deg = list(np.sort(np.random.randint(low=0, high=360, size=(num_endpoints))))
        endpoints_deg.append(endpoints_deg[0])
        distance_deg = []
        for i in range(num_endpoints-1):
            distance_deg.append(endpoints_deg[i+1]-endpoints_deg[i])
        distance_deg.append(360 - endpoints_deg[-1])
        if np.sort(distance_deg)[0] > (180./num_endpoints):
            break
    endpoints = angles_to_point(endpoints_deg, square_size, end_pt_offset)

    curve = []
    for icurve in range(num_endpoints):
        points = []
        points.append(tuple(endpoints[icurve]))
        for icp in range(num_ctrl_points):
            ctrl_pt = sample_coord(square_size, control_pt_offset)
            points.append(tuple(ctrl_pt))
        points.append(tuple(endpoints[icurve+1]))
        points_supersampled = supersample(points, ss)
        bezier = make_bezier(list(points_supersampled))
        curve += bezier(ts)

    #draw.polygon(points, outline='black')
    draw.line(curve, fill='black', width=thickness*ss)
    im = im.resize([square_size, square_size], resample=Image.BILINEAR)
    array = np.array(im)
    # DOWNSAMPLE
    im.save('out.png')
    return array


if __name__ == '__main__':
    square_size = 200
    num_endpoints = 3
    num_ctrl_points = 1
    num_lines = 50
    end_pt_offset = 5
    control_pt_offset =20
    thickness = 4
    ss = 5
    generate_item(square_size, num_endpoints, num_ctrl_points, end_pt_offset, control_pt_offset, num_lines, thickness, ss)