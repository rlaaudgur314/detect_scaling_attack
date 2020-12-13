import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim

def rescaling(image, size): # use cv2
    height, width, channel = image.shape

    scaled_img = cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)
    rescaled_img = cv2.resize(scaled_img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    return rescaled_img


def filtering(image): # use PIL
    filtered_img = image.filter(ImageFilter.MinFilter(size=5))

    return filtered_img


def mse(image1, image2):
    height, width, channel = image1.shape

    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(height * width)

    return error


f_normal_rescaling_MSE  = open("normal_rescaling_MSE.txt", 'w')
f_normal_filtering_MSE  = open("normal_filtering_MSE.txt", 'w')
f_attack_rescaling_MSE  = open("attack_rescaling_MSE.txt", 'w')
f_attack_filtering_MSE  = open("attack_filtering_MSE.txt", 'w')
f_normal_rescaling_SSIM = open("normal_rescaling_SSIM.txt", 'w')
f_normal_filtering_SSIM = open("normal_filtering_SSIM.txt", 'w')
f_attack_rescaling_SSIM = open("attack_rescaling_SSIM.txt", 'w')
f_attack_filtering_SSIM = open("attack_filtering_SSIM.txt", 'w')

for i in range(1, 101):
	# attack image
	dir_path = "./attack_1"
	img_path = dir_path + "/%03d.jpg"%i

	img_PIL = Image.open(img_path)
	img_PIL_filtering = filtering(img_PIL)
	img_cv2_filtering = cv2.cvtColor(np.array(img_PIL_filtering), cv2.COLOR_RGB2BGR)

	img_cv2 = cv2.imread(img_path)
	target_size = (64, 64)
	img_cv2_rescaling = rescaling(img_cv2, target_size)

	m_rescaling = mse(img_cv2, img_cv2_rescaling)
	s_rescaling = ssim(img_cv2, img_cv2_rescaling, multichannel=True)
	m_filtering = mse(img_cv2, img_cv2_filtering)
	s_filtering = ssim(img_cv2, img_cv2_filtering, multichannel=True)

	f_attack_rescaling_MSE.write(str(m_rescaling)+'\n')
	f_attack_rescaling_SSIM.write(str(s_rescaling)+'\n')
	f_attack_filtering_MSE.write(str(m_filtering)+'\n')
	f_attack_filtering_SSIM.write(str(s_filtering)+'\n')


	# normal image
	dir_path = "./sample_1"
	img_path = dir_path + "/%03d.jpg"%i

	img_PIL = Image.open(img_path)
	img_PIL_filtering = filtering(img_PIL)
	img_cv2_filtering = cv2.cvtColor(np.array(img_PIL_filtering), cv2.COLOR_RGB2BGR)

	img_cv2 = cv2.imread(img_path)
	target_size = (64, 64)
	img_cv2_rescaling = rescaling(img_cv2, target_size)

	m_rescaling = mse(img_cv2, img_cv2_rescaling)
	s_rescaling = ssim(img_cv2, img_cv2_rescaling, multichannel=True)
	m_filtering = mse(img_cv2, img_cv2_filtering)
	s_filtering = ssim(img_cv2, img_cv2_filtering, multichannel=True)
	
	f_normal_rescaling_MSE.write(str(m_rescaling)+'\n')
	f_normal_rescaling_SSIM.write(str(s_rescaling)+'\n')
	f_normal_filtering_MSE.write(str(m_filtering)+'\n')
	f_normal_filtering_SSIM.write(str(s_filtering)+'\n')

f_normal_rescaling_MSE.close()
f_normal_filtering_MSE.close()
f_attack_rescaling_MSE.close()
f_attack_filtering_MSE.close()
f_normal_rescaling_SSIM.close()
f_normal_filtering_SSIM.close()
f_attack_rescaling_SSIM.close()
f_attack_filtering_SSIM.close()

