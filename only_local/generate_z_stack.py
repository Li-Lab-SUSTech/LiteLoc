import numpy as np
import csv

image_size = 200
num_points = 2
center_size = int(image_size * 0.85)
z_stack_range = np.linspace(start=3000, stop=-3000, num=60)
x, y, z, frame, intensity = [], [], [], [], []
pixel_size = [108, 108]
frame_count = 1

for i in z_stack_range:
    frame_ = np.repeat(frame_count, num_points)
    x_center = np.array([6000, 12000])
    y_center = np.array([6000, 12000])
    z_ = np.repeat(i, num_points)
    intensity_ = np.random.uniform(40000, 60000, num_points)

    x.append(x_center)
    y.append(y_center)
    z.append(z_)
    frame.append(frame_)
    intensity.append(intensity_)

    frame_count = frame_count + 1

x, y, z, frame, intensity = (np.array(x).reshape(num_points*60, ), np.array(y).reshape(num_points*60, ),
                  np.array(z).reshape(num_points*60, ), np.array(frame).reshape(num_points*60, ), np.array(intensity).reshape(num_points*60, ),)
ground_truth = np.linspace(start=1, stop=num_points*60, num=num_points*60)

with open('z_stack_tetra6.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['ground_truth', 'frame', 'X', 'Y', 'Z', 'intensity'])
    for row in zip(ground_truth, frame, x, y, z, intensity):
        csvwriter.writerow(row)