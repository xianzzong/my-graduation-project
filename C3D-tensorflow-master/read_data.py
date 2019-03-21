import numpy as np
import cv2


def main():
    data = np.load('test.npy')
    B, N, W, H, C = data.shape
    for b in xrange(B):
        for n in xrange(N):
            cv2.imwrite('test.png', data[b][n])
    return


if __name__ == '__main__':
    main()
