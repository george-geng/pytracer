import PIL
import math
from PIL import Image
import numpy as np
import time
import numbers
from math import sqrt, tan, pi
import matplotlib.pyplot as plt
#Set up image dimensions here
imageWidth = 640
imageHeight = 480
INFINITY = 10.e20
MAXDEPTH = 2
bias = 1.e-4

class vec3():
	def __init__(self, x=0, y=0, z=0):
		self.x = x
		self.y = y
		self.z = z
	def __str__(self):
		return "vec3({}, {}, {})".format(*self)
	def __add__(self, v):
		return vec3(self.x + v.x, self.y + v.y, self.z + v.z)
	def __mul__(self, v):
		if isinstance(v, numbers.Number):
			return vec3(self.x*v, self.y*v, self.z*v)
		elif isinstance(v, vec3):
			return self.x*v.x + self.y*v.y + self.z*v.z
	def __rmul__(self, v):
		return self.__mul__(v)
	def __sub__(self, v):
		return vec3(self.x - v.x, self.y - v.y, self.z - v.z)
	def __neg__(self):
		return vec3(-self.x, -self.y, -self.z)
	def __div__(self, v):
		if v is not 0:
			return vec3(self.x/v, self.y/v, self.z/v)
	def __truediv__(self, v):
		return vec3(self.x/v, self.y/v, self.z/v)
	def __iter__(self):
		yield self.x
		yield self.y
		yield self.z
	def dot(self, v):
		return self.x*v.x + self.y*v.y + self.z*v.z
	def cross(self, v):
		return vec3(self.y*v.z-self.z*v.y, self.z*v.x - self.x*v.z, self.x*v.y - self.y*v.x)
	def norm(self):
		return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
	def normalize(self):
		return self/self.norm()
	def reflect(self, v):
		v = v.normalize()
		return self - 2 * (self * v) * v
	def asTuple(self):
		return (self.x, self.y, self.z)
	def clip(self, lower, upper):
		if (self.x < lower): self.x = lower
		if (self.x > upper): self.x = upper
		if (self.y < lower): self.y = lower
		if (self.y > upper): self.y = upper
		if (self.z < lower): self.z = lower
		if (self.z > upper): self.z = upper
		return vec3(self.x, self.y, self.z)
#just have one light for now..that would be great already 
globalLight = vec3(5, 5., -10) #this will be a point light
lightCol = vec3(1.,1.,1.)

class Ray():
	def __init__(self, ro, rd):
		self.o = ro
		self.d = rd

class Sphere(): 
	def __init__(self, center, radius, diffuseColor, mirror = 0.5):
		self.c = center;
		self.r = radius;
		self.diffuse = diffuseColor;
		self.mirror = mirror;

	def getDiffuse(self, hitPoint):
		return self.diffuse

	def intersect (self, ray):
		#print(ray.d)
		#print(ray.o-self.c)
		a = ray.d*ray.d
		oc = ray.o - self.c
		b = 2*ray.d.dot(oc)
		c = oc*oc-self.r*self.r
		#c = self.c*self.c + ray.o*(ray.o) - 2*self.c*(ray.o) - self.r*self.r
		disc = b*b - 4*a*c
		if disc < 0:
			#print('NO GOOD')
			return INFINITY
		q = (-b -math.sqrt(disc))/2. if b < 0 else (-b + math.sqrt(disc))/2.
		t0 = q/a
		t1 = c/q
		t0,t1 = min(t0,t1), max(t0,t1)
		if t1>=0:
			#print('hit somthing')
			return t1 if t0 < 0 else t0
		return INFINITY
		#return t

class CheckeredSphere(Sphere):
	def getDiffuse(self, hitPoint):
		if ((int)(hitPoint.x*2))%2 == ((int)(hitPoint.z*2))%2:
			return self.diffuse
		else: return vec3(0,0,0)

class Plane():
	def __init__(self, point, normal, diffuseColor, mirror = 0.5):
		self.p = point
		self.n = normal
		self.diffuse = diffuseColor
		self.mirror = mirror

	def getDiffuse(self,hitPoint):
		if (int((hitPoint.x * 2))%2) == (int((hitPoint.z * 2))%2):
			return self.diffuse
		else: 
			return vec3(0,0,0)

	def intersect(self, ray):
		den = ray.d.dot(self.n) 
		#if den < 0: print(den)
		if (abs(den) < 1e-6): #the ray/plane are parallel
			return INFINITY
	#	print(self.p)
	#	print(ray.o)
		t = (self.p-ray.o).dot(self.n)/den
		#return t
		if t < 0 :
			#print(t, den)
			return INFINITY
		#print('hit a plane')
		return t

def getColor(obj, r, t, objs, numHits):
	isectP = r.o + r.d*t;
	isectN = None
	if isinstance (obj,Sphere):
		isectN = (isectP - obj.c).normalize()
	elif isinstance (obj, Plane):
		isectN = obj.n.normalize()
	lightDir = (globalLight-isectP).normalize()
	eyeDir = (eye - isectP).normalize()
	#adjusted to avoid shadow acne
	adjustP = isectP + isectN*bias
	shadowRay = Ray(adjustP, lightDir)
	#cast a shadow ray
	shadowRayIsects = []
	for ob in objs:
		if ob is not obj:
			shadowRayIsects.append(ob.intersect(shadowRay))
	canSee = 1.0;
	if len(shadowRayIsects) > 0:
		closestIsect = min(shadowRayIsects)
		if closestIsect < INFINITY:
			canSee = 0.0;
	#if (visible): canSee = 1.0;

	#ambient color
	color = vec3(0.05,0.05,0.05) 
	#lambertian shade
	cosTheta = max(0,isectN.dot(lightDir))
	#print(color)
	#print(obj.diffuse)
	color = color + (obj.getDiffuse(isectP) *cosTheta * canSee)
	#reflection
	if numHits < MAXDEPTH:
		newDir = r.d.reflect(isectN).normalize()
		color = color + castRay(Ray(adjustP, newDir), objs, numHits+1)*obj.mirror;
	#specular shading
	phong = max(0,isectN.dot((lightDir + eyeDir).normalize()).clip(0,1))**50*lightCol
	color = color + phong; 

	return color
def castRay(r, objs, numHits = 0):
	#we want to find the nearest intersection point
	tnear = INFINITY
	surfaceCol = vec3(0,0,0)
	
	#intersections = []
	for i, obj in enumerate(objs):
		tCurr =(obj.intersect(r))
		if tCurr < tnear:
			tnear = tCurr
			objIndex = i

	if tnear == INFINITY: 
		#print("black background")
		return vec3(0,0,0) #background col
	objNear = objs[objIndex]

	return getColor(objNear, r, tnear, objs, numHits)
	#print("diff color")
	



#generate image below
asp = float(imageWidth)/imageHeight
fov = 30.
angle = math.tan(math.pi * 0.5 * fov /180)
eye = vec3(0,0.35,-1.)
testSphere = Sphere(vec3(.75, .1, 1.), .6, vec3(0,0,1))
sphere2 = Sphere(vec3(-.75, .1, 2.25), .6, vec3(.5, .223, .5))
sphere3 = Sphere(vec3(-2.75, .1, 3.5), .6, vec3(1., .572, .184))

checkerBoard = Plane(vec3(0.,-.5,0.), vec3(0.,1.,0.), vec3(1.,1.,1.))

checkerBall = CheckeredSphere(vec3(0,-99999.5,0), 99999, vec3(.75,.75,.75),.25)
scene = [testSphere, sphere2, sphere3, checkerBoard]



w = imageWidth
h = imageHeight
img = np.zeros((h, w, 3))
S = (-1., -1. / asp + .25, 1., 1. / asp + .25)
xCoords = np.linspace(S[0], S[2], w)
yCoords = np.linspace(S[1], S[3], h)

#img = Image.new("RGB", (w,h))
for i,x in enumerate(xCoords):
#	print('i:' + str(i))
	for j,y in enumerate(yCoords):
#		print('j' + str(j))
		col = vec3(0,0,0)
		castDir = (vec3(x, y, 0.)-eye).normalize();
		col += castRay(Ray(eye, castDir), scene)
		colArray = np.array([col.x,col.y,col.z])
		img[h-j-1, i, :] = np.clip(colArray,0,1)

plt.imsave('fig1.png', img)

# img = Image.new("RGB", (imageWidth, imageHeight))
# for x in range(imageWidth):
# 	print(x)
# 	for y in range(imageHeight):
# 		dirX = 2*(x + 0.5)/(imageWidth-1) * asp * angle
# 		dirY = (1-2*(y + 0.5))/(imageHeight-2) * angle
# 		rayDir = vec3(dirX, dirY, -1).normalize()
# 	#	rayDir = (vec3(x/50.0-5, y/50.0-5,0)-eye).normalize()
# 		col = trace(Ray(eye, rayDir), scene)
# 		img.putpixel((x, 479-y), (col.asTuple()))
#img.save("trace4.png","PNG")




