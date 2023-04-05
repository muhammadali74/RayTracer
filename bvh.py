import numpy as np
from math import *
# from raytracer_premature import Sphere


class BoundingBox:
    def __init__(self, objects):
        self.bounds = self.compute_bounds(objects)
        minx = self.bounds[0][0]
        miny = self.bounds[0][1]
        minz = self.bounds[0][2]
        maxx = self.bounds[1][0]
        maxy = self.bounds[1][1]
        maxz = self.bounds[1][2]
        self.coordinates = ([(minx, miny, minz), (minx, maxy, minz), (minx, maxy, maxz), (minx, miny, maxz),
                            (maxx, maxy, maxz), (maxx, miny, maxz), (maxx, miny, minz), (maxx, maxy, minz)])
        self.center = self.compute_center(self.coordinates)
        self.sphere = None

        if type(objects) != list:
            self.left, self.right = None, None
            self.sphere = objects
        else:
            self.left = objects[0]
            if len(objects) > 1:
                self.right = objects[1]
            else:
                self.right = None

    def compute_center(self, coordinates):
        center = np.zeros(3)
        for coord in coordinates:
            center += coord
        return center/8

    def compute_bounds(self, objects):
        if type(objects) != list:
            return objects.bounds
        else:
            min_bounds = np.min([obj.bounds[0] for obj in objects], axis=0)
            max_bounds = np.max([obj.bounds[1] for obj in objects], axis=0)

            return (min_bounds, max_bounds)

    def __str__(self) -> str:
        return str([self.center, str(self.left), str(self.right)])

    def box_intersect(self, rayOrigin, rayDirection):
        t0 = (self.bounds[0] - rayOrigin) / rayDirection
        t1 = (self.bounds[1] - rayOrigin) / rayDirection
        # tmin = np.maximum(tmin, np.zeros_like(tmin))
        # tmax = np.minimum(tmax, np.ones_like(tmax) * np.inf)
        # return np.all(tmin <= tmax)

        tmin = np.min([t0, t1], axis=0)
        tmax = np.max([t0, t1], axis=0)

        return np.max(tmin) <= np.min(tmax)


class BVHTree:
    def __init__(self, objects):
        self.root = None
        self.boxes = [BoundingBox(obj) for obj in objects]

    def __str__(self):
        lst = []
        for box in self.boxes:
            lst.append(str(box))
        return str(lst)

    def buildBVH(self):
        while len(self.boxes) > 1:
            best = inf
            left, right = None, None
            for i in range(len(self.boxes)):
                for j in range(i+1, len(self.boxes)):
                    dist = self.compute_dist(self.boxes[i], self.boxes[j])
                    if dist < best:
                        best = dist
                        left = i
                        right = j

            new_box = BoundingBox([self.boxes[left], self.boxes[right]])
            del self.boxes[right]
            del self.boxes[left]
            self.boxes.append(new_box)

    def buildTree(self):
        self.buildBVH()
        self.root = self.boxes[0]
        # return self.root

    def compute_dist(self, box1, box2):
        center1 = box1.center
        center2 = box2.center
        dist = sqrt((center1[0] - center2[0])**2 + (center1[1] -
                    center2[1])**2 + (center1[2] - center2[2])**2)

        return dist

    def traverse(self, rayOrigin, rayDirection):
        stack = [self.root]
        minDist = inf
        nearest = None
        while len(stack) > 0:
            node = stack.pop()
            # print(node)
            if node.box_intersect(rayOrigin, rayDirection):
                if node.left is None and node.right is None:
                    dist = node.sphere.intersect(rayOrigin, rayDirection)
                    # print(dist)
                    if dist is not None:
                        if dist < minDist:
                            minDist = dist
                            nearest = (node.sphere, dist)

                else:
                    stack.append(node.left)
                    stack.append(node.right)

        if nearest is None:
            return None, None
        return nearest
