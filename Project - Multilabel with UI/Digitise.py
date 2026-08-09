import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = r"C:\Users\mmudo\OneDrive\Dokumenter\Supercurricular\Chemistry Neural Net\Project - Multilabel\example.png"

def calibrate_axis(image_path):
    #pick 4 points on the graph for extracting data (like cropping the image)
    
    title_string = "Click the 4 corners of your image"
    subtitle_string = "x-axis at wavenumber 4000, x-axis at wavenumber 400, y-axis at transmittance 0%, y-axis at transmittance 100%"

    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"{title_string}\n{subtitle_string}")

    points = plt.ginput(4, timeout = 0)
    plt.close(fig)

    return points

def build_calibration(points, x_values, y_values):
    #points[0] and points[1] are x-axis, points[2] and points[3] are y-axis
    #x and y values are the desired values whereas points are the coordinates on the plot before calibration
    (x1_px, _), (x2_px, _) = points[0], points[1]
    (_, y1_px), (_, y2_px) = points[2], points[3]

    x1_val, x2_val = x_values
    y1_val, y2_val = y_values

    def pixel_to_x(px): #linear interpolation to place a given pixel
        frac = (px-x1_px) / (x2_px - x1_px)
        return x1_val + frac*(x2_val - x1_val)

    def pixel_to_y(py):
            frac = (py-y1_px) / (y2_px - y1_px)
            return y1_val + frac*(y2_val - y1_val)

    return pixel_to_x, pixel_to_y

def extract_curve(image_path, threshold=200):
    img = cv2.imread(image_path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY_INV)

    height, width = binary.shape
    curve_points = []
    for x in range(width):
        column = binary[:,x]
        dark_pixels = np.where(column > 0)[0]
        if len(dark_pixels) > 0:
            y = int(np.mean(dark_pixels))
            curve_points.append((x,y))

    return curve_points

def digitise_spectrum(image_path, x_values, y_values):
    points = calibrate_axis(image_path)
    pixel_to_x, pixel_to_y = build_calibration(points, x_values, y_values)

    curve_points = extract_curve(image_path)

    x_data = np.array([pixel_to_x(px) for px, py in curve_points])
    y_data = np.array([pixel_to_y(py) for px, py in curve_points])

    return x_data, y_data