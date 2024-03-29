import numpy as np
import matplotlib.pyplot as plt

from bvh import *


# import numpy as np
# from math import *
# # from raytracer_premature import Sphere


# class BoundingBox:
#     def __init__(self, objects):
#         self.bounds = self.compute_bounds(objects)
#         minx = self.bounds[0][0]
#         miny = self.bounds[0][1]
#         minz = self.bounds[0][2]
#         maxx = self.bounds[1][0]
#         maxy = self.bounds[1][1]
#         maxz = self.bounds[1][2]
#         self.coordinates = ([(minx, miny, minz), (minx, maxy, minz), (minx, maxy, maxz), (minx, miny, maxz),
#                             (maxx, maxy, maxz), (maxx, miny, maxz), (maxx, miny, minz), (maxx, maxy, minz)])
#         self.center = self.compute_center(self.coordinates)
#         self.sphere = None

#         if type(objects) != list:
#             self.left, self.right = None, None
#             self.sphere = objects
#         else:
#             self.left = objects[0]
#             if len(objects) > 1:
#                 self.right = objects[1]
#             else:
#                 self.right = None

#     def compute_center(self, coordinates):
#         center = np.zeros(3)
#         for coord in coordinates:
#             center += coord
#         return center/8

#     def compute_bounds(self, objects):
#         if type(objects) != list:
#             return objects.bounds
#         else:
#             min_bounds = np.min([obj.bounds[0] for obj in objects], axis=0)
#             max_bounds = np.max([obj.bounds[1] for obj in objects], axis=0)

#             return (min_bounds, max_bounds)

#     def __str__(self) -> str:
#         return str([self.center, str(self.left), str(self.right)])

#     def box_intersect(self, rayOrigin, rayDirection):
#         t0 = (self.bounds[0] - rayOrigin) / rayDirection
#         t1 = (self.bounds[1] - rayOrigin) / rayDirection
#         # tmin = np.maximum(tmin, np.zeros_like(tmin))
#         # tmax = np.minimum(tmax, np.ones_like(tmax) * np.inf)
#         # return np.all(tmin <= tmax)

#         tmin = np.min([t0, t1], axis=0)
#         tmax = np.max([t0, t1], axis=0)

#         return np.max(tmin) <= np.min(tmax)


# class BVHTree:
#     def __init__(self, objects):
#         self.root = None
#         self.boxes = [BoundingBox(obj) for obj in objects]

#     def __str__(self):
#         lst = []
#         for box in self.boxes:
#             lst.append(str(box))
#         return str(lst)

#     def buildBVH(self):
#         while len(self.boxes) > 1:
#             best = inf
#             left, right = None, None
#             for i in range(len(self.boxes)):
#                 for j in range(i+1, len(self.boxes)):
#                     dist = self.compute_dist(self.boxes[i], self.boxes[j])
#                     if dist < best:
#                         best = dist
#                         left = i
#                         right = j

#             new_box = BoundingBox([self.boxes[left], self.boxes[right]])
#             del self.boxes[right]
#             del self.boxes[left]
#             self.boxes.append(new_box)

#     def buildTree(self):
#         self.buildBVH()
#         self.root = self.boxes[0]
#         # return self.root

#     def compute_dist(self, box1, box2):
#         center1 = box1.center
#         center2 = box2.center
#         dist = sqrt((center1[0] - center2[0])**2 + (center1[1] -
#                     center2[1])**2 + (center1[2] - center2[2])**2)
class BVHTree:
    def __init__(self, objects):
        self.root = None
        self.boxes = [BoundingBox(obj) for obj in objects]  #list of bounding boxes for all the objects in the scene

    def __str__(self):
        lst = []
        for box in self.boxes:
            lst.append(str(box))  #list of strings of all the bounding boxes in the scene for checking purposes
        return str(lst)

    #Took this approach from the research paper (Agglomerative clustering for bounding volume hierarchies)
    def buildBVH(self):
        while len(self.boxes) > 1:
            best = inf      #best is the distance between the two closest boxes
            left, right = None, None       #left and right are the indices of the two closest boxes
            for i in range(len(self.boxes)):      #iterate over all the boxes
                for j in range(i+1, len(self.boxes)):     #iterate over all the boxes after the ith box 
                    dist = self.compute_dist(self.boxes[i], self.boxes[j])     #compute the distance between the two boxes
                    if dist < best:    #if the distance is less than the best distance, then update the best distance and the indices of the two closest boxes
                        #Yahan pe tree banta jaa raha hai, har node k baad jo closest box hai usko left child banata jaa raha hai aur jo dusra closest box hai usko right child banata jaa raha hai
                        best = dist  
                        left = i      
                        right = j

            #YEH WALA PART SAARE BOXES KO COMBINE KARKE NAYA BOX BANATA HAI JISMEIN DONO CHILDREN HAI IS TARAH ITTERATIVELY KRTA HAI TILL THERE IS ONLY ONE BOX LEFT JO K SCENE KA BOUNDING BOX HOGA
            new_box = BoundingBox([self.boxes[left], self.boxes[right]]) #create a new box with the two closest boxes as its children and combine 
            del self.boxes[right]  #delete the right box from the list of boxes
            del self.boxes[left]    #delete the left box from the list of boxes
            self.boxes.append(new_box)  #append the new box to the list of boxes 
            #IS TARAH TREE CHOTA HUTA CHALA JAEGA FROM THE BOTTOM TILL THERE IS ONLY ONE BOX LEFT

    def buildTree(self):
        self.buildBVH()
        self.root = self.boxes[0]  #the root of the tree will be the only box left in the list of boxes
        # return self.root

    def compute_dist(self, box1, box2):
        center1 = box1.center            #the center of the first box 
        center2 = box2.center         #the center of the second box
        dist = sqrt((center1[0] - center2[0])**2 + (center1[1] -
                    center2[1])**2 + (center1[2] - center2[2])**2) #compute the distance between the two centers by distance formula

#         return dist

    def traverse(self, rayOrigin, rayDirection):
        stack = [self.root]    #stack to traverse the tree initially the root is pushed in the stack
        minDist = inf   #minimum distance of the intersection point from the ray origin
        nearest = None  #nearest object to the ray origin
        while len(stack) > 0:
            node = stack.pop()  #pop the top element from the stack
            # print(node)
            if node.box_intersect(rayOrigin, rayDirection):  #check if the ray intersects the box
                if node.left is None and node.right is None:  #if the node is a leaf node 
                    dist = node.sphere.intersect(rayOrigin, rayDirection)    #check if the ray intersects the sphere
                    # print(dist)
                    if dist is not None:    #if the ray intersects the sphere
                        if dist < minDist:  #if the distance of the intersection point from the ray origin is less than the minimum distance
                            minDist = dist  #update the minimum distance
                            nearest = (node.sphere, dist)  #update the nearest object to the ray origin

                else:
                    stack.append(node.left)  #push the left child of the node in the stack
                    stack.append(node.right)    #push the right child of the node in the stack

#         if nearest is None:
#             return None, None
#         return nearest




























###############NECHE POORI LINEAR ALGEBRA HAI JISSE HAMNE SPHERE INTERSECTION KA CODE LIKHA HAI#####################
class Sphere(object):
    def __init__(self, center, radius, ambient, diffuse, specular, shininess, reflection):
        self.center = center
        self.radius = radius
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.bounds = np.array([center - radius, center + radius])  #bounding box of the sphere uper and lower bounds

    def intersect(self,  ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - self.center)   #doubling the ray direction and then dotting it with the ray origin minus the center of the sphere
        c = np.linalg.norm(ray_origin - self.center) ** 2 - self.radius ** 2  #calculating norm and then squaring it and then subtracting the radius squared from it
        delta = b ** 2 - 4 * c   #calculating the discriminant
        if delta > 0:     #if the discriminant is greater than 0	
            t1 = (-b + np.sqrt(delta)) / 2  #calculating the first intersection point
            t2 = (-b - np.sqrt(delta)) / 2  #calculating the second intersection point
            if t1 > 0 and t2 > 0:
                return min(t1, t2)  #return the minimum of the two intersection points
        return None


class Light(object):
    def __init__(self, position, ambient, diffuse, specular):
        self.position = position
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular


def normalize(vector):
    return vector / np.linalg.norm(vector)     #normalizing the vector


def reflected(vector, axis):   
    return vector - 2 * np.dot(vector, axis) * axis     #calculating the reflected vector


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

#########YEH KIA KR RAHA HAI?##############
def nearest_intersected_object(objects, ray_origin, ray_direction): #function to find the nearest object that the ray intersects
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


#yeh GUI ka code hai
width = 300
height = 200

max_depth = 1

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom


light = Light(np.array([5, 5, 5]), np.array([1, 1, 1]),
              np.array([1, 1, 1]), np.array([1, 1, 1]))  #light source ka position, ambient, diffuse and specular color

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
]   #list of objects in the scene with their center, radius, ambient, diffuse, specular color, shininess and reflectivity

tree = BVHTree(objects)
print(tree)
tree.buildTree()
print(tree)
# tree.remove(objects[5])
print(tree)

newsp = Sphere(np.array([-0.1, 0.1, -1]), 0.6, np.array([0.3, 0.3, 0.3]),
           np.array([0, 0.1, 0]), np.array([1, 1, 1]), 100, 0.4)
tree.insert(newsp)

image = np.zeros((height, width, 3))
# plt.show()


#The code below is for the ray tracing algorithm and the GUI part
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))

        # nearest_object, min_distance = nearest_intersected_object(
        #     objects, origin, direction)
        nearest_object, min_distance = tree.traverse(origin, direction)
        if nearest_object is None:

            continue
        else:
            # color = nearest_object['color']
            # print('isNOTNone')

            intersection = origin + min_distance * direction
            normal_to_surface = normalize(
                intersection - nearest_object.center)
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(
                light.position - shifted_point)

            # _, min_distance = nearest_intersected_object(
            #     objects, shifted_point, intersection_to_light)
            # intersection_to_light_distance = np.linalg.norm(
            #     light.position - intersection)
            # is_shadowed = min_distance < intersection_to_light_distance

            # if is_shadowed:
            #     break

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
            color += illumination

            # color = nearest_object['color']

        image[i, j] = np.clip(color, 0, 1)

    print("%d/%d" % (i + 1, height))
    # myobj.set_data(image)
    # plt.show()


# plt.imsave('imagenew.png', image)
plt.imshow(image)
plt.show()
