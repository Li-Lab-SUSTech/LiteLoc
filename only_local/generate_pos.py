from utils.help_utils import generate_pos


image_size = 256
pixel_size = [108, 108]
num_points = 40000
z_scale = 700
save_path = "/home/feiyue/liteloc_git/pos4w_size256.csv"

generate_pos(image_size, pixel_size, num_points, z_scale, save_path)