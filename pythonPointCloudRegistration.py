import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import open3d as o3d
import copy

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)
state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

def concatPointCloud(cloud1, cloud2):
	cloud1Points = np.asarray(cloud1.points)
	cloud2Points = np.asarray(cloud2.points)
	
	cloud1Colors = np.asarray(cloud1.colors)
	cloud2Colors = np.asarray(cloud2.colors)
	
	cloud3Points = np.concatenate((cloud1Points, cloud2Points))
	cloud3Colors = np.concatenate((cloud1Colors, cloud2Colors))
	
	cloud3 = o3d.geometry.PointCloud()
	cloud3.points = o3d.utility.Vector3dVector(cloud3Points)
	cloud3.colors = o3d.utility.Vector3dVector(cloud3Colors)
	return cloud3

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

def draw_registration_result_original_color(source, target, transformation):
	source_temp = copy.deepcopy(source)
	source_temp.transform(transformation)
	o3d.visualization.draw_geometries([source_temp, target], zoom=0.5, front=[-0.2458, -0.8088, 0.5342], lookat=[1.7745, 2.2305, 0.9787], up=[0.3109, -0.5878, -0.7468])
def pointCloudRegistration(source, target):
	threshold = 10
	current_transformation = np.identity(4)
	result_icp = o3d.pipelines.registration.evaluate_registration(source, target, threshold, current_transformation)
	print(result_icp)
	current_transformation = result_icp.transformation
	return current_transformation

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def coloredPointCloudRegistration(source, target):
	#voxel_radius = [0.04, 0.02, 0.01]
	#max_iter = [50, 30, 14]
	#max_iter = [1, 1, 1]
	voxel_radius = [0.5, 0.25, 0.125]
	max_iter = [50, 30, 14]
	current_transformation = np.identity(4)
	for scale in range(3):
		iter = max_iter[scale]
		radius = voxel_radius[scale]
		print([iter, radius, scale])

		print("3-1. Downsample with a voxel size %.2f" % radius)
		source_down = source.voxel_down_sample(radius)
		target_down = target.voxel_down_sample(radius)

		print("3-2. Estimate normal.")
		source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2,max_nn=30))
		target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2,max_nn=30))


		print("3-3. Applying colored point cloud registration")
		result_icp = o3d.pipelines.registration.registration_colored_icp(
	        source_down, target_down, radius, current_transformation,
	        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
	        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
		current_transformation = result_icp.transformation
		print(result_icp)

	return current_transformation

out = np.empty((h, w, 3), dtype=np.uint8)

priorVerts = np.array([])
priorTexcoords = np.array([])
colors = np.array([])
transform = np.identity(4)
pcd = o3d.geometry.PointCloud()
pcdPrior = o3d.geometry.PointCloud()
threshold = 0.2
#threshold = 2
#for i in range(1):
notFirstIteration = False
while True:
	#time.sleep(5)
	if not state.paused:
		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()

		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		
		#print("color_frame")
		#print(type(color_frame))
		#print(dir(color_frame))

		depth_frame = decimate.process(depth_frame)

		# Grab new intrinsics (may be changed by decimation)
		depth_intrinsics = rs.video_stream_profile(
		    depth_frame.profile).get_intrinsics()
		w, h = depth_intrinsics.width, depth_intrinsics.height

		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

		if state.color:
		    mapped_frame, color_source = color_frame, color_image
		else:
		    mapped_frame, color_source = depth_frame, depth_colormap

		points = pc.calculate(depth_frame)
		pc.map_to(mapped_frame)

		# Pointcloud data to arrays
		v, t = points.get_vertices(), points.get_texture_coordinates()
		verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
		texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
		
		#print("verticies")
		#print(verts.shape)
		
		#print("color_image")
		#print(color_image.shape)
		#print(color_image)
		
		#print("color_source")
		#print(color_source.shape)
		
		#print("points")
		#print(points.size())
		
		#print("depth_colormap")
		#print(depth_colormap.shape)
		
		#print("state.color")
		#print(state.color)
		
		#print("color_frame")
		#print(color_frame)
		
		#print("depth_image")
		#print(depth_image.shape)
		
		#print("color_image")
		#print(color_image.shape)
		#print(color_image[0][0])
		
		#print("texcoords")
		#print(texcoords.shape)
		#for t in texcoords:
		#	print(texcoords.argmax())
		#	print(texcoords.argmin())
		#	print(texcoords.flatten()[texcoords.argmin()])
		#	print(texcoords.flatten()[texcoords.argmax()])
		#	print("texcoord: " + str(t))
		
		#ok, so color_image looks suspiciously like a numpy array cast of the color frame which seems to contain the texture data. I'm going to try and normalize the texture coordinates to work with it and see where that goes. https://github.com/IntelRealSense/librealsense/issues/6234 this maps from texture coordinates to color frame
		colorsRGB = np.zeros((len(texcoords), 3))
		#print("colorsRGB: " + str(colorsRGB.shape))
		#idx = np.zeros(len(verts))
		#print()
		w = color_image.shape[1] #480
		h = color_image.shape[0] #640
		#print("w: " + str(w))
		#print("h: " + str(h))
		#color_array = color_image.flatten()/255
		color_image = color_image/255
		#print("color_array length: " + str(len(color_array)))
		for i, t in enumerate(verts):
			#coordX = np.rint(t[0] * color_image.shape[0])
			#coordY = np.rint(t[1] * color_image.shape[1])
			#colorLocation = int(coordY * color_image.shape[1] + coordX)
			#colorsRGB[i] = [int(color_image.flatten()[colorLocation]), int(color_image.flatten()[colorLocation + 1]), int(color_image.flatten()[colorLocation + 2])]
			
			#print("texcoords x: " + str(texcoords[i][0]))
			#print("texcoords y: " + str(texcoords[i][1]))
			#print("texcoords: " + str(texcoords[i]))
			
			x = min(max(int(texcoords[i][0]*w + 0.5),0),w - 1)
			#print("x: " + str(x))
			y = min(max(int(texcoords[i][1]*h + 0.5),0),h - 1)
			#print("y: " + str(y))
			#print("bytes per pixel: " + str(color_frame.get_bytes_per_pixel()))
			#print("stride in bytes: " + str(color_frame.get_stride_in_bytes()))
			#idx = int(x * color_frame.get_bytes_per_pixel() / 8 + y * color_frame.get_stride_in_bytes())
			#idx = int(x + y*w + i * 3)
			#idx = x + y * w
			#idx = x + y
			#print(idx) #921620
			
			#colorsRGB[i] = [color_array[idx], color_array[idx + 1], color_array[idx + 2]]
			colorsRGB[i] = color_image[y][x]
			#print("coordX: " + str(coordX))
			#print("coordY: " + str(coordY))
			#colorLocation = int(coordY * color_image.shape[1] + coordX)
			#print("colorLocation: " + str(colorLocation))
			#print([color_image.flatten()[colorLocation], color_image.flatten()[colorLocation + 1], color_image.flatten()[colorLocation + 2]])
			
			#colorsRGB[i] = [int(color_image.flatten()[colorLocation]), int(color_image.flatten()[colorLocation + 1]), int(color_image.flatten()[colorLocation + 2])]
			#print(colorsRGB[i])
			#colorsRGB[i] = [0.5, 0.5, 0.5]
			
			#print(i)
			#colorsRGB = np.concatenate(([color_image.flatten()[colorLocation], color_image.flatten()[colorLocation + 1], color_image.flatten()[colorLocation + 2]], colorsRGB))
			#np.concatenate([color_image.flatten()[colorLocation], color_image.flatten()[colorLocation + 1], color_image.flatten()[colorLocation + 2]], colorsRGB)
		#print("idx max: " + str(idx.max())) #1227059
		#print("idx max indicy? " + str(idx[int(idx.max())]))
		#print("i: " + str(i))
		#print("colorRGB shape:")
		#print(colorsRGB.shape)
		#print(colorsRGB[i])
		#print("color_array max: " + str(color_array.max()))
		#print("color_array min: " + str(color_array.min()))
		#print("color_array mean: " + str(color_array.mean()))
		#print("texcoords.shape:")
		#print(texcoords.shape)
		
		
		#mapped frame looks to be the texture based on
		
		#for i, j in enumerate(verts):
			#print("yo")
		#	print(color_source.shape)
		#	print(color_source)
		#	print(texcoords[i])
		#	print(texcoords[i][0])
		#	print(texcoords[i][1])
		#	print(color_source[texcoords[i][0]][texcoords[i][1]])
			#color_source[]
			#np.append(colors, color_source[])
		
		#texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 3)  # uv
		#texcoords cannot be converted to o3d cloud color. Why?
		#needs to be array of 3x1 arrays filled with rgb values
		#currently is... 76800x2 array of something?
		#vert is 76800x3 which makes sense. Where is third color in texcoords?
		#so its a uv map of the texture, not the actual colors
		#how to get colors from librealsense
		
		#mytexcoords = np.asarray(t)
		#print(type(texcoords))
		#print(dir(texcoords))
		#print(mytexcoords.shape)
		#print("print first element of mytexcoords")
		#print(mytexcoords[1000])
		#print(texcoords.shape)
		#print(verts.shape)
		#print(texcoords)
		#print(t)
		#print("color_source")
		#print(color_source.shape)
		#print(color_source[10][10])
		
		#I'm just going to test saving as .ply then loading back in to o3d
		#points.export_to_ply('./Test.ply', mapped_frame)
		#input_file = "Test.ply"
		#pcdTest = o3d.io.read_point_cloud(input_file) # Read the point cloud
		
		#points.get_texcolor()
		
		#verts = np.asarray(pcdTest.points)
		#colorsTest = np.asarray(pcdTest.colors)
		#print(colorsTest)
		#print(colorsTest.shape)
		#Now it has the rgb format. let me look into how the librealsense outputs to .ply

	#add prior verts to current verts
	#if priorVerts.any():
	
	#apparently o3d might be able to take in depth and uv map directly according to function create_from_rgbd_image
	#http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
	pcd.points = o3d.utility.Vector3dVector(verts)
	pcd.colors = o3d.utility.Vector3dVector(colorsRGB)
	
	#for some reason, icp functions will not work. Apparently I can change things like voxel downsample and radius outliers and have some kind of effect in the pointcloud?
	#pcd = pcd.voxel_down_sample(0.0000001)
	#print(dir(pcd))
	#print("pcd min bound: " + str(pcd.get_min_bound()))
	#print("pcd max bound: " + str(pcd.get_max_bound()))
	#pcd.remove_radius_outlier(5,1)
	
	if notFirstIteration:
		#pcd.points = o3d.utility.Vector3dVector(verts)
		#pcd.colors = o3d.utility.Vector3dVector(colorsRGB)
		#pcdPrior.points = o3d.utility.Vector3dVector(priorVerts)
		#pcdPrior.colors = o3d.utility.Vector3dVector(priorTexcoords)
		
		#verts = np.concatenate((verts, priorVerts), axis=0)
		#texcoords = np.concatenate((colorsTest, priorTexcoords), axis=0)
		
		#print(verts.shape)
		
		#result = o3d.pipelines.registration.registration_icp(pcd, pcdPrior, threshold, transform,o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
		
		result = o3d.pipelines.registration.registration_icp(pcd, pcdPrior, threshold, transform,o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
		
		transform = result.transformation
		print(transform)
		
		pcd.transform(transform)
		pcd = concatPointCloud(pcd, pcdPrior)
		pcd = pcd.voxel_down_sample(0.00001)
		#draw_registration_result(pcd, pcdPrior, transform)
		o3d.visualization.draw_geometries([pcd])
		#o3d.visualization.draw_geometries([pcd])
		#o3d.visualization.draw_geometries([pcdPrior])
		
		
		#transform = coloredPointCloudRegistration(pcd, pcdPrior)
		#print("colored icp")
		#print(transform)
		#transform = pointCloudRegistration(pcd, pcdPrior)
		#print("icp")
		#print(transform)

	#Registration requires o3d pointcloud object. need to convert np ndarrays
	#steps
	#must take old and new point cloud and convert to o3d clouds
	#pass both clouds to colored point cloud registration function above
	#passes back transform from one to the other
	#apply transform to new pointcloud (numpy?) and concat them together.
	#display to see if it worked
	#use technique to even out density of data (cull copy points so as not to eat up all resources)
	
	# Render
	#now = time.time()

	#out.fill(0)

	#grid(out, (0, 0.5, 1), size=1, n=10)
	#frustum(out, depth_intrinsics)
	#axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

	#if not state.scale or out.shape[:2] == (h, w):
	#	pointcloud(out, verts, texcoords, color_source)
	#else:
	#	tmp = np.zeros((h, w, 3), dtype=np.uint8)
	#	pointcloud(tmp, verts, texcoords, color_source)
	#	tmp = cv2.resize(
	#	tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
	#	np.putmask(out, tmp > 0, tmp)

	#if any(state.mouse_btns):
	#	axes(out, view(state.pivot), state.rotation, thickness=4)

	#dt = time.time() - now

	#cv2.setWindowTitle(
	#	state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
	#	(w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

	#cv2.imshow(state.WIN_NAME, out)
	#key = cv2.waitKey(1)
	
	#o3d.visualization.draw_geometries([pcd])
	#key = cv2.waitKey(1)
	
	#input current pointcloud into old pointcloud
	#priorPoints = points
	#priorVerts = verts
	#priorTexcoords = colorsTest
	
	#after running demo, turns out icp needs original point cloud to be close to new cloud. Theshold controls how precise this is. It was being made up for by the trans_init matrix they gave. With an identity matrix, we can see how far away this icp method works. threshold had to be raised to 0.2. Lets see if it will work on my point clouds in similar conditions.
	
	#demo_icp_pcds = o3d.data.DemoICPPointClouds()
	#source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
	#target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
	
	#print("source min bound: " + str(source.get_min_bound()))
	#print("source max bound: " + str(source.get_max_bound()))
	
	#threshold = 0.2
	#trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
        #                 [-0.139, 0.967, -0.215, 0.7],
        #                 [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
	#trans_init = np.identity(4)
	#reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
	
	#reg_p2p = o3d.pipelines.registration(np.identity(4))
	#iden = np.identity(4)
	
	#draw_registration_result(source, target, reg_p2p.transformation)


	
	pcdPrior = copy.deepcopy(pcd)
	notFirstIteration = True
	
pipeline.stop()
	
