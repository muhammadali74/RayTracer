import numpy as np
from math import *
# from raytracer_premature import Sphere
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

    def __eq__(self, node):
        # if node.minx == self.minx and node.miny == self.miny and node.minz == self.minz and node.maxx == self.maxx and node.maxy == self.maxy and node.maxz == self.maxz:
        if id(node) == id(self):    
            return True
        return False

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
        cdef float best
        cdef unsigned long long int i
        cdef unsigned long long int j
        cdef unsigned long long int left
        cdef unsigned long long int right

        while len(self.boxes) > 1:
            best = inf
            left, right = 0, 0
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

        cdef float minDist

        # if type == 'origin':
        stack = [self.root]
        minDist = inf
        nearest = None
        while len(stack) > 0:
            node = stack.pop()
            # print(node)
            if node.box_intersect(rayOrigin, rayDirection):
                if node.left is None and node.right is None:
                    dist = node.sphere.intersect(rayOrigin, rayDirection)

                    if dist is not None:
                        if dist < minDist:
                            minDist = dist
                            nearest = (node.sphere, dist)

                else:
                    # if node.left is not None:
                    stack.append(node.left)
                    stack.append(node.right)

        if nearest is None:
            return None, inf
        return nearest

        # else:

    def remove(self, obj):
        stack = [(self.root, self.root.left), (self.root, self.root.right)]
        # pa = []
        while len(stack) > 0:
            sett = stack.pop()
            parent = sett[0]
            node = sett[1]

            if node.left is None and node.right is None:
                if node.sphere == obj:
                    node.sphere = None

                    if parent == self.root:
                        if parent.left == node:
                            self.root = parent.right
                            return
                        elif parent.right == node:
                            self.root = parent.left
                            return



                    grand_p = self.find(parent)

                    if parent.left == node:
                        if grand_p.left == parent:
                            grand_p.left = parent.right
                        elif grand_p.right == parent:
                            grand_p.right = parent.right
                    elif parent.right == node:
                        if grand_p.left == parent:
                            grand_p.left = parent.left
                        elif grand_p.right == parent:
                            grand_p.right = parent.left


            else:
                stack.append((node,node.left))
                stack.append((node,node.right))
        
        return "not found"
    
    def find(self, node):
        stack = [(self.root, self.root.left), (self.root, self.root.right)]
        while len(stack) > 0:
            n = stack.pop()
            if n[1] == node:
                return n[0]
            else:
                if n[1].left is not None:
                    stack.append(n[1].left)
                if n[1].right is not None:
                    stack.append(n[1].right)
        return None
    
    def insert(self, obj):
        x = BoundingBox(obj)
        parents = []
        dis = []
        stack = [self.root]
        minDist = inf
        while len(stack) > 0:
            node = stack.pop()
            parents.append((node,node.left,node.right))
            if node.left is None and node.right is None:
                dist = self.compute_dist(node,x)
                if dist < minDist:
                    minDist = dist
                    dis.append(node)
            else:
                stack.append(node.left)
                stack.append(node.right)

        node = dis[-1]
        q = BoundingBox([node,x])

        for i in parents:
            if i[1] == node:
                i[0].left = q
            if i[2] == node:
                i[0].right = q