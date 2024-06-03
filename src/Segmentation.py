import numpy as np
import cv2
from skimage.morphology import skeletonize, thin


class Segmentation:

    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
    
    def get_ends(self, skelet):
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(skelet, cv2.CV_32F, kernel).astype(int)
        return np.argwhere((skelet > 0) & (neighbor_count == 1))


    def prune_thinned(self, thinned, min_length=5):
        pruned = np.copy(thinned)
        for i in range(min_length):
            # Find endpoints
            endpoints = (cv2.filter2D(pruned.astype(np.uint8), -1, np.ones((3, 3))) == 2) & pruned
            pruned[endpoints] = 0
        return pruned


    def process_image1(self):
        new_image = self.image.copy()

        new_image[..., 2][(new_image[..., 2] > 78)] = 0
        new_image[..., 1][(new_image[..., 1] > 56)] = 0
        new_image[..., 0][(new_image[..., 0] > 43)] = 0

        image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        image_gray[image_gray > 0] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        image_gray = cv2.dilate(image_gray, kernel)
        image_gray = cv2.dilate(image_gray, kernel)
        image_gray = cv2.medianBlur(image_gray, ksize=11)

        num_labels, labels = cv2.connectedComponents(image_gray)

        max_comp = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
        image_gray[labels != max_comp] = 0

        image_gray = cv2.erode(image_gray, kernel)

        skelet_image = image_gray.copy()
        skelet_image[skelet_image > 0] = 1
        skeleton = skeletonize(skelet_image)
        skeleton = skeleton.astype(np.uint8)
        skeleton[skeleton == 1] = 255

        thinned = thin(skeleton)

        pruned_thinned = self.prune_thinned(thinned, min_length=5)
        for i in range(4):
            pruned_thinned = self.prune_thinned(pruned_thinned, min_length=5)
        pruned_thinned = pruned_thinned.astype(np.uint8)
        pruned_thinned[pruned_thinned > 0] = 255
        return pruned_thinned


    def process_image2(self, pruned_thinned):
        img = self.image.copy()

        num_labels, labels = cv2.connectedComponents(pruned_thinned)
        max_comp = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
        pruned_thinned[labels != max_comp] = 0

        img[..., 0][img[..., 0] > 10] = 0
        img[..., 1][img[..., 1] > 10] = 0
        img[..., 2][img[..., 2] > 5] = 0

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        kernel_dilate2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, kernel_erode)
        gray = cv2.dilate(gray, kernel_dilate)
        gray = cv2.medianBlur(gray, ksize=17)
        gray[gray > 0] = 255

        pruned_thinned[pruned_thinned > 0] = 1
        endpoints = self.get_ends(pruned_thinned)
        pruned_thinned[pruned_thinned > 0] = 255

        one_points = np.zeros_like(pruned_thinned)
        for endpoint in endpoints:
            cv2.circle(one_points, tuple(endpoint[::-1]), 5, 255, -1)

        one_points = cv2.dilate(one_points, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        gray2 = cv2.dilate(gray, kernel_dilate2)

        gray1 = one_points + gray
        gray1[gray1 > 0] = 1

        gray2 = gray2 + cv2.dilate(one_points, kernel_dilate2)
        gray2[gray2 > 0] = 1

        gray2 = gray2 - gray1

        gray2[gray2 > 0] = 255

        res = [0, 0, 0, 0, 0]
        num_labels, labels = cv2.connectedComponents(gray2)

        for i in range(1, num_labels):
            mask = (labels == i).astype(np.uint8)
            deg, _ = cv2.connectedComponents(cv2.bitwise_and(pruned_thinned, pruned_thinned, mask=mask))

            if deg > 1:
                if deg - 2 < len(res):
                    res[deg - 2] += 1
                else:
                    res.extend([0] * ((deg - 1) - len(res)))
                    res[deg - 2] += 1

        return res, pruned_thinned


# process_image2(process_image1())[0]
