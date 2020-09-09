import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate,ndimage
import cv2

#team_members_names = ['باسل سعيد محمد','اسماء علاء عبد الحى','آية سعيد محمد','اسامة امجد لحظى','اسلام حسين سلام']
#team_members_seatnumbers = ['2016170082','2015170058','2014170141','2016170091','2015170109']

def hough_line(img):
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(int(-diag_len), int(diag_len), int(diag_len) * 2)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * int(diag_len), num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + round(diag_len))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

def draw_lines_connected(img, lines, color=[255, 0, 0], thickness=8):
    #this function should draw lines to the images (default color is red and thickness is 8)

    return


def convert_rbg_to_grayscale(img):
    # This function will do color transform from RGB to Gray
    grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return grayImage


def convert_rgb_to_hsv(img):
    # This function will do color transform from RGB to HSV
    hsvImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsvImage

def detect_edges_canny(img, low_threshold, high_threshold):
    # You should implement yoru Canny Edge Detector here
    old_img = img
    (G, theta) = sobel_filters(img)
    img = non_max_suppression(G, theta)

    (img, weak, strong) = threshold(img, low_threshold, high_threshold)
    img = hysteresis(img, weak, strong)
    # img2 = old_img - img
    img2 = img
    return img2


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # angle 0
            if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                q = img[i, j + 1]
                r = img[i, j - 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = img[i + 1, j]
                r = img[i - 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

    return Z


def threshold(img, lowThreshold, highThreshold):
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong) or (
                        img[i, j - 1] == strong) or (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong) or (
                        img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def remove_noise(img, kernel_size):
    pixels = np.asarray(img)
    h = len(img)
    w = len(img[1])
    gaussian = np.zeros([kernel_size, kernel_size])
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian[i, j] = (np.exp(-(i ** 2 + j ** 2) / 2 * (1.1 ** 2)) / (2 * np.pi * (1.1 ** 2)))

    kernel = gaussian
    img = ndimage.filters.convolve(img, kernel)
    # x = 0
    # y = 0
    # offset = kernel_size // 2
    # for x in range(1, h - offset):
    #     for y in range(1, w - offset):
    #         acc = 0
    #         pixel = pixels[x - offset:(x + kernel_size - offset), y - offset:(y + kernel_size - offset)]
    #         out = np.multiply(pixel, kernel)
    #         # for a in range(kernel_size):
    #         #     for b in range(kernel_size):
    #         #         xn = x + a - offset
    #         #         yn = y + b - offset
    #         #         pixel = pixels[xn, yn]
    #         #         acc += pixel * kernel[a, b]
    #         img[x, y] = np.sum(out)

    return img
    
def mask_image(img, vertices):
    #Mask out the pixels outside the region defined in vertices (set the color to black)
    return


#main part
def main():
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('out.avi', fourcc, 30, (540, 960))
    vidcap = cv2.VideoCapture('SolidYellowTest.mp4')
    success, image = vidcap.read()
    while True:
        ret, frame = vidcap.read()
        cv2.imwrite("frame.jpeg",frame)
        if ret == True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv_image = convert_rgb_to_hsv(frame)
        gray_image = convert_rbg_to_grayscale(frame)

        sensitivity = 20
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        lower_yellow = np.array([45, 110, 110])
        upper_yellow = np.array([240, 250, 250])
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        res = mask_white | mask_yellow
        res[0:350, :] = 0
        # res[:, 0:50] = 0
        # res[:, 900:960] = 0
        gray_image[0:350, :] = 0
        # gray_image[:, 0:50] = 0
        # gray_image[:, 900:960] = 0
        h, w = res.shape
        # plt.imshow(mask_yellow)
        # plt.show()
        gray_image = gray_image & res
        # plt.imshow(gray_image)
        # plt.show()
        gray_image = remove_noise(gray_image, 5)
        gray_image = detect_edges_canny(gray_image, 75, 150)
        # plt.imshow(gray_image)
        # plt.show()
        accumulator, thetas, rhos = hough_line(gray_image)
        threshold = accumulator.max() * 0.1
        rhos, thetas = np.where(accumulator > threshold)

        T = np.arange(-90, 90)
        width, height = gray_image.shape
        diag_len = np.ceil(np.sqrt(width * width + height * height))
        im = np.zeros((width, height, 3), dtype=np.int32)
        posPointsx1 = []
        posPointsx2 = []
        posPointsy1 = []
        posPointsy2 = []

        negPointsx1 = []
        negPointsx2 = []
        negPointsy1 = []
        negPointsy2 = []

        for i, j in zip(rhos, thetas):
            theta = np.deg2rad(T[j])
            rho = i - diag_len

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0:
                posPointsx1.append(x1)
                posPointsx2.append(x2)
                posPointsy1.append(y1)
                posPointsy2.append(y2)


            else:
                negPointsx1.append(x1)
                negPointsx2.append(x2)
                negPointsy1.append(y1)
                negPointsy2.append(y2)

        x1p = x2p = y1p = y2p = 0
        x1n = x2n = y1n = y2n = 0

        for i in range(len(posPointsx1)):

            x1p += posPointsx1[i]
            x2p += posPointsx2[i]
            y1p += posPointsy1[i]
            y2p += posPointsy2[i]


        for i in range(len(negPointsx1)):

            x1n += negPointsx1[i]
            x2n += negPointsx2[i]
            y1n += negPointsy1[i]
            y2n += negPointsy2[i]

        x1p /= len(posPointsx1)
        x2p /= len(posPointsx1)
        y1p /= len(posPointsx1)
        y2p /= len(posPointsx1)

        x1n /= len(negPointsx1)
        x2n /= len(negPointsx1)
        y1n /= len(negPointsx1)
        y2n /= len(negPointsx1)


        cv2.line(frame, (int(x1p), int(y1p)), (int(x2p), int(y2p)), (0, 0, 255), 10)
        cv2.line(frame, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (0, 0, 255), 10)


        cv2.imwrite("output.jpeg", frame)
        cv2.imshow('frame', frame)
    vidcap.release()
    out.release()
    cv2.destroyAllWindows()



main()

#1 read the image
#2 convert to HSV
#3 convert to Gray
#4 Threshold HSV for Yellow and White (combine the two results together)
#5 Mask the gray image using the threshold output fro step 4
#6 Apply noise remove (gaussian) to the masked gray image
#7 use canny detector and fine tune the thresholds (low and high values)
#8 mask the image using the canny detector output
#9 apply hough transform to find the lanes
#10 apply the pipeline you developed to the challenge videos

#11 You should submit your code
