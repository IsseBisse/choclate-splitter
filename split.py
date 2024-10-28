import cv2
import numpy as np
import random

image = cv2.imread("chocolate_mask.png")
mask = np.ones_like(image[:, :, 0])

outline_color = np.array([76, 177, 34])

for colorband in range(3):
    mask &= image[:, :, colorband] == outline_color[colorband]
    
cv2.imwrite("mask.png", mask*255)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

output = np.zeros_like(image)
contour_stats = list()
for idx, contour in enumerate(contours):
    color = [random.randint(50, 255) for _ in range(3)]
    cv2.drawContours(output, [contour], -1, color, thickness=cv2.FILLED)

    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    area = cv2.contourArea(contour)
    # First contour will be removed so we decrement idx by one
    cv2.putText(output, str(idx-1), (cX-50, cY+50), cv2.FONT_HERSHEY_PLAIN, 5, 0, 2)

    contour_stats.append({"center": (cX, cY), "area": area})
    
# First contour == entire outline. Remove it!
contour_stats = contour_stats[1:]

def distribute_values(values, n_people):
    sorted_values = sorted(values, reverse=True)
    
    people = [[] for _ in range(n_people)]
    sums = [0] * n_people
    
    for idx, value in enumerate(sorted_values):
        min_sum_index = sums.index(min(sums))
        people[min_sum_index].append(idx)
        sums[min_sum_index] += value
    
    percentage_sums = [sub_sum/sum(sums) for sub_sum in sums]

    return people, percentage_sums

def pretty_print(sets, sums):
    for idx, sub_set in enumerate(sets):
        print(f"Person {idx} ({sums[idx]*100:.1f} %): {', '.join([str(num) for num in sub_set])}")
    print("")

for n_people in range(2, 5):
    sets, sums = distribute_values([cnt["area"] for cnt in contour_stats], n_people)

    print(f"=== Distribution with {n_people} people ===")
    pretty_print(sets, sums)

cv2.imwrite("split.png", output)
