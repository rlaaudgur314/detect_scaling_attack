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


filtering_MSE_T = 8200
filtering_SSIM_T = 0.35
rescaling_MSE_T = 6000
rescaling_SSIM_T = 0.42

filtering_MSE_acc_normal = 0
filtering_SSIM_acc_normal = 0
rescaling_MSE_acc_normal = 0
rescaling_SSIM_acc_normal = 0
filtering_MSE_acc_attack = 0
filtering_SSIM_acc_attack = 0
rescaling_MSE_acc_attack = 0
rescaling_SSIM_acc_attack = 0

for i in range(1, 101):
	# test attack image
	dir_path = "./attack_2"
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
	
	if(m_rescaling >= rescaling_MSE_T):
		rescaling_MSE_acc_attack+=1
	if(s_rescaling <= rescaling_SSIM_T):
		rescaling_SSIM_acc_attack+=1
	if(m_filtering >= filtering_MSE_T):
		filtering_MSE_acc_attack+=1
	if(s_filtering <= filtering_SSIM_T):
		filtering_SSIM_acc_attack+=1


	# test normal image
	dir_path = "./sample_2"
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
	
	if(m_rescaling < rescaling_MSE_T):
		rescaling_MSE_acc_normal+=1
	if(s_rescaling > rescaling_SSIM_T):
		rescaling_SSIM_acc_normal+=1
	if(m_filtering < filtering_MSE_T):
		filtering_MSE_acc_normal+=1
	if(s_filtering > filtering_SSIM_T):
		filtering_SSIM_acc_normal+=1


print("[Rescaling - MSE]")
print("normal : %d, attack : %d"%(rescaling_MSE_acc_normal ,rescaling_MSE_acc_attack))
print("[Rescaling - SSIM]")
print("normal : %d, attack : %d"%(rescaling_SSIM_acc_normal ,rescaling_SSIM_acc_attack))
print("[Filtering - MSE]")
print("normal : %d, attack : %d"%(filtering_MSE_acc_normal ,filtering_MSE_acc_attack))
print("[Filtering - SSIM]")
print("normal : %d, attack : %d"%(filtering_SSIM_acc_normal ,filtering_SSIM_acc_attack))


