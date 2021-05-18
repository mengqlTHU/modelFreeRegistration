import open3d as o3d
import numpy as np
import copy

pc = o3d.io.read_point_cloud("./x64/Release/Cropped_Frame120.pcd")
keypoints = o3d.io.read_point_cloud("./x64/Release/keypoints.pcd")

vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
pc.paint_uniform_color([1, 1, 1])
keypoints.paint_uniform_color([1, 0, 0])

vis.add_geometry(pc)
vis.add_geometry(keypoints)
vis.run()


