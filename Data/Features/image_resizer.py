from PIL import Image
import torchvision.transforms as transforms
from os.path import join, exists
from os import listdir, makedirs
from shutil import rmtree


def resize_ds_images(images_path, dest_path):
	transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), ])
	it_cls = 1
	total_cls = len(listdir(images_path))
	for f in listdir(images_path):
		new_folder_path = join(dest_path, f)
		if exists(new_folder_path):
			rmtree(new_folder_path)
		makedirs(new_folder_path)
			
		folder_path = join(images_path, f)
		images = listdir(join(images_path, f))
		total_images = len(images)
		it = 1

		for img_name in images:
			image = Image.open(join(folder_path, img_name))
			image = transform(image)
			image.save(join(new_folder_path, img_name))
			print('class {} - {}/{}, image {}/{} done'.format(f, it_cls, total_cls, it, total_images))
			it += 1
		it_cls += 1	

images_path = '/data/cvg/hamed/Datasets/AWA2/images_original'
dest_path = '/data/cvg/hamed/Datasets/AWA2/images'

#resize_ds_images(images_path, dest_path)