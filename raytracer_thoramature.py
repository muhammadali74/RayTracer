import numpy as np
import matplotlib.pyplot as plt
import random
from bvh import *

import time


class Sphere(object):
    def __init__(self, center, radius, ambient, diffuse, specular, shininess, reflection):
        self.center = center
        self.radius = radius
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.bounds = np.array([center - radius, center + radius])

    def intersect(self,  ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - self.center)
        c = np.linalg.norm(ray_origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None


class Light(object):
    def __init__(self, position, ambient, diffuse, specular):
        self.position = position
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular


def normalize(vector):
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


# naive approach to find the ray intersection.
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(
        obj.center, obj.radius, ray_origin, ray_direction) for obj in objects]
    # print(distances)
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


width = 300
height = 200

max_depth = 1

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom


light = Light(np.array([5, 5, 5]), np.array([1, 1, 1]),
              np.array([1, 1, 1]), np.array([1, 1, 1]))

objects = [
    Sphere(np.array([-0.2, 0, -1]), 0.7, np.array([0.1, 0, 0]),
           np.array([0.7, 0, 0]), np.array([1, 1, 1]), 100, 0.5),
    Sphere(np.array([0.1, -0.3, 0]),  0.1, np.array([0.1, 0, 0.1]),
           np.array([0.7, 0, 0.7]),  np.array([1, 1, 1]),  100, 0.5),
    Sphere(np.array([-0.3, 0, 0]), 0.15, np.array([0, 0.1, 0]),
           np.array([0, 0.6, 0]), np.array([1, 1, 1]), 100, 0.5),
    Sphere(np.array([-0.5, 0.5, -0.5]), 0.15, np.array([0, 0.1, 0.6]),
           np.array([0, 0.6, 0]), np.array([1, 1, 1]), 100, 0.5),
    Sphere(np.array([-0.3, 0.4, -2]), 0.9, np.array([0.2, 0.1, 0]),
           np.array([0, 0.6, 0]), np.array([1, 1, 1]), 100, 0.5),
    Sphere(np.array([0, -9000, 0]), 9000 - 0.7, np.array([0.1, 0.1, 0.1]),
           np.array([0.6, 0.6, 0.6]), np.array([1, 1, 1]), 100, 0.5),
]

newobjects = [Sphere(np.array([random.uniform(-0.3, 0.7), random.uniform(0, 0.8), random.uniform(0, -3)]), random.uniform(0, 0.2), np.array([0.1, 0.1, 0.1]),
                     np.array([random.uniform(0, 0.6), random.uniform(0, 0.6), random.uniform(0, 0.6)]), np.array([1, 1, 1]), 100, random.uniform(0, 0.5)) for k in range(300)]

objects.extend(newobjects)

tree = BVHTree(objects)
print(tree)
tree.buildTree()
print(tree)

image = np.zeros((height, width, 3))
# plt.show()
start = time.time()

for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            # nearest_object, min_distance = tree.traverse(origin, direction)
            nearest_object, min_distance = nearest_intersected_object(
                objects, origin, direction)

            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = normalize(
                intersection - nearest_object.center)
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(
                light.position - shifted_point)

            _, min_distance = nearest_intersected_object(
                objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(
                light.position - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                break

            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object.ambient * light.ambient

            # diffuse
            illumination += nearest_object.diffuse * light.diffuse * \
                np.dot(intersection_to_light, normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object.specular * light.specular * np.dot(
                normal_to_surface, H) ** (nearest_object.shininess / 4)

            # reflection
            color += reflection * illumination
            reflection *= nearest_object.reflection

            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)

    print("%d/%d" % (i + 1, height))
    # myobj.set_data(image)
    # plt.show()


end = time.time()
print(end - start)


image = np.zeros((height, width, 3))
# plt.show()
start = time.time()

for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            nearest_object, min_distance = tree.traverse(origin, direction)
            # nearest_object, min_distance = nearest_intersected_object(
            #     objects, origin, direction)

            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = normalize(
                intersection - nearest_object.center)
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(
                light.position - shifted_point)

            # _, min_distance = nearest_intersected_object(
            #     objects, shifted_point, intersection_to_light)
            _, min_distance = tree.traverse(
                shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(
                light.position - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                break

            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object.ambient * light.ambient

            # diffuse
            illumination += nearest_object.diffuse * light.diffuse * \
                np.dot(intersection_to_light, normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object.specular * light.specular * np.dot(
                normal_to_surface, H) ** (nearest_object.shininess / 4)

            # reflection
            color += reflection * illumination
            reflection *= nearest_object.reflection

            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)

    print("%d/%d" % (i + 1, height))
    # myobj.set_data(image)
    # plt.show()


end = time.time()
print(end - start)


plt.imsave('imagenew.png', image)
plt.imshow(image)
plt.show()
