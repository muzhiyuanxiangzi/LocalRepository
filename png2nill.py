from PIL import Image
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join
import cv2


def singlenii2png():
    png_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble\\'
    nii_path = 'D:\\StudyProgram\\Program\\VScode\\Dataset\\300Coronary\\RCA105\\RCA_labe_nii\\Forgeground\\'
    save_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble_png\\'
    nii1 = sitk.ReadImage(nii_path)
    nii2 = sitk.GetArrayFromImage(nii1)
    png = Image.fromarray(nii2)
    png.save(png_path + 'test.png')


def singlepng2nii():
    png_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble_nonebk\\1.png'
    nii_path = 'D:\\StudyProgram\\Program\\VScode\\Dataset\\300Coronary\\RCA105\\RCA_labe_nii\\Forgeground\\'
    save_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble_nonebk\\'
    nii1 = sitk.ReadImage(png_path)
    nii2 = sitk.GetArrayFromImage(nii1)
    nii = sitk.GetImageFromArray(nii2)
    sitk.WriteImage(nii,save_path + 'test.nii')
# singlepng2nii()


def custom_sort_key(list):
    number = []
    # for x in list:
    # item = list.split('.nii')[0]
    item = list.split('.')[0]
    number.append(item)
    item = int(item)#将listdir返回的字符串转换成int类型，这样才能比较大小
    return item

def nii2png():
    scribble_background_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble\\'
    scribble_forgeground_path = 'D:\\StudyProgram\\Program\\VScode\\Dataset\\300Coronary\\RCA105\\RCA_labe_nii\\Forgeground\\'
    save_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble_png\\'
    file_names = [x for x in sorted(listdir(join(scribble_background_path)), key=custom_sort_key)]

    # 使用 sorted 函数进行自定义排序
    # sorted_list = sorted(my_list, key=custom_sort_key)
    # file_names2 = [x for x in sorted(listdir(join(scribble_path)), key=custom_sort_key)]
    i = 0
    for file_name in file_names:
        x = file_names[i].split('.')[0]
        nii_bk = sitk.ReadImage(os.path.join(scribble_background_path + file_name) )
        nii_bk = sitk.GetArrayFromImage(nii_bk)#.astype('uint8')
        # nii_fg = sitk.ReadImage(os.path.join(scribble_forgeground_path + file_name) )
        # nii_fg = sitk.GetArrayFromImage(nii_fg)#.astype('uint16')
        # png2nii = sitk.GetImageFromArray(nii2)
        # nii_all = nii_bk + nii_fg
        if nii_bk.shape[0] == 1:
            nii = nii_bk[0]
        elif nii_bk.shape[2] == 3:
            # nii_bk = np.squeeze(nii_bk, axis=2)
            nii = nii_bk[:,:,1]#这里的1是测试出来的，即把三个通道的数值都提取出来看最大值有没有变化，可以写个if判断句
            if nii.max() != 2:
                print(file_name,'选1通道出现问题')
                nii = nii_bk[:,:,2]
                if nii.max() != 2:
                    print(file_name,'选2通道出现问题,最终选择0通道')
                    nii = nii_bk[:,:,0]
                    if nii.max() != 2:
                        print(file_name,'选0通道出现问题')
                        
        nii2png = Image.fromarray(nii)
        nii2png.save(save_path + x + '.png')
        i = i + 1
        # sitk.WriteImage(png2nii,scribble_path + x + '.nii')
    print('转换成功')

# nii2png()
def png2nii(nii_path,png_path):
    # ###sitk包读取png文件,并和Image读取的数据进行对比  使用sitk读取指定文件夹下的若干png文件并保存成nii文件
    temp_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\Scribble\\225.nii.gz'
    temp_path2 = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\Niitype\\'

    nii = sitk.ReadImage(temp_path)
    nii = sitk.GetArrayFromImage(nii).astype('uint8')
    nii[nii == 1] = 255
    # nii = nii.transpose((2, 0, 1))
    png = Image.fromarray(nii)
    png.save('225.png')
    png2nii = sitk.GetImageFromArray(nii)
    sitk.WriteImage(png2nii, '18552_23.nii')
    file_names = [x for x in listdir(join(temp_path))]
    for file_name in file_names:
        x = file_name.split('.')[0]
        nii1 = sitk.ReadImage(os.path.join(temp_path + file_name) )
        nii2 = sitk.GetArrayFromImage(nii1)
        png2nii = sitk.GetImageFromArray(nii2)
        sitk.WriteImage(png2nii,temp_path2 + x + '.nii')
    print('转换成功')

def png2png():
    # ###sitk包读取png文件,并和Image读取的数据进行对比  使用sitk读取指定文件夹下的若干png文件并保存成nii文件
    temp_path = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\Scribble_nonebk\\'
    temp_path2 = 'D:\\StudyProgram\\Program\\VScode\\Process_json\\arcade_dataset_phase_1\\segmentation_dataset\\seg_train\\Main679\\train\\add_Image\\'

    # nii = sitk.ReadImage(temp_path)
    # nii = sitk.GetArrayFromImage(nii)#.astype('uint8')

    # file_names = [x for x in sorted(listdir(join(temp_path)), key=custom_sort_key)]
    file_names = [x for x in listdir(join(temp_path))]
    for file_name in file_names:
        x = file_name.split('.')[0]
        nii1 = sitk.ReadImage(os.path.join(temp_path + file_name) )
        nii2 = sitk.GetArrayFromImage(nii1)
        # nii2 = np.where(nii2 == 2, 0)
        nii2[nii2 == 2] = 0
        png2png = Image.fromarray(nii2)
        png2png.save(temp_path + x + '.png')
        # png2nii = sitk.GetImageFromArray(nii2)
        # sitk.WriteImage(png2nii,temp_path2 + x + '.nii')
    print('转换成功')
png2png()

# png1 = Image.open(png_path)
# png2 = np.array(png1)
# print(png2)
# # plt.imsave('999.jpg',nii2)
# image_nii = sitk.GetImageFromArray(nii2)
# sitk.WriteImage(image_nii,'/public/home/shenjl/Pythoncode/DSCNet-main/444_png.nii')

# nii2.reshape(1,224,224)
# nii2 = nii2[0:224,0:224]

# print(nii2,nii2.shape)
# image_nii = Image.fromarray(nii2)
# image_nii.save('777.png')
# img1 = Image.open(png_path)
# img2 = np.array(img1)
# print(img2,nii2)
# print(img2-nii2)
# print("---")
# image_nii = sitk.GetImageFromArray(nii2)
# sitk.WriteImage(image_nii,'/public/home/shenjl/Pythoncode/DSCNet-main/222.nii')



# empt_mat.append(img2)

# emp=np.array(empt_mat)
# nii_file = sitk.GetImageFromArray(emp)
# # 此处的emp的格式为样本数*高度*宽度*通道数
# #不要颠倒这些维度的顺序，否则文件保存错误
# sitk.WriteImage(nii_file,nii_path) # nii_path 为保存路径
    


#关于nii图片，其数值是在零一之间分布的，各种数据都有；nii标签则是数值上只有零或者一，图片和标签的数据类型都是浮点型，换成整型的话ITK软件就打不开
#关于png图片，首先其数值类型基本上都是uint8，然后是在0~255之间分布，也是各种数值都有，label的话是只有0和255这两个数值的。


##生成随机矩阵
# # img = np.zeros((224,224))
# img = np.random.randint(low=0, high=2, size=(224, 224))*255
# # img = np.random.rand(size=(224, 224))*100
# print(img)
# img = np.array(img).astype(dtype=np.uint8)
# # img = img.reshape(224,224) 
# print(img)

##读取npy文件
# for i in range(0,10):
#     np.save(str(i)+'.npy',img)
#     # i = i + 1
# np.save('666.npy',img)
# npy_path = '/public/home/shenjl/Pythoncode/DSCNet-main/19.npy'
# npy_file = np.load(npy_path)
# print(npy_file)

# img_real = Image.fromarray(((npy_file)*255).astype(dtype=np.uint8))
# img_real.save('19.png')

##利用Image读取png文件
# empt_mat=[]
# # for i in png_path:
# img1=Image.open(png_path)#.convert('RGB')
# img2 = np.array(img1)
# # png_nii = sitk.GetImageFromArray(img2)
# # sitk.WriteImage(png_nii,'444.nii')
# img2=np.array(img1).astype(np.float32)
# # img2.reshape(224,224)
# img2 = img2[0:224,0:224]
# print(img2,img2.shape)
# img2 = (img2-1).astype(dtype='uint8')
# img2 = img2 + 1
# img3 = np.array(img1)
# img3 = img3[0:224,0:224]
# image = Image.fromarray(img2)#.convert('RGB')
# image.save('111.png')

##读取npy文件并将其保存成jpg
# npy_path = '/public/home/shenjl/Pythoncode/DSCNet-main/DSCNet/0.npy'
# npy_file = np.load(npy_path)
# print(npy_file)
# npy_file = npy_file*255
# print(npy_file)
# plt.imsave('888.jpg', npy_file)

##将png的label映射到零一之间，以及进一步转换成零一分布，再保存成nii文件看看是什么样
# image = Image.open(png_path).convert('L')
# image1 = np.array(image)
# img = Image.fromarray(image1)
# img.save('3channel.png')
# print(image1)
# image1 = image1/np.max(image1)
# print(image1)
# image1_nii = sitk.GetImageFromArray(image1)
# sitk.WriteImage(image1_nii,'image1_nii.nii')
# image2 = (np.where(image1>0.5,1,0)).astype(dtype=np.float64)
# image2_nii = sitk.GetImageFromArray(image2)
# sitk.WriteImage(image2_nii,'image2_nii.nii')
# image_sub = image2-image1
# print(image_sub)

# ##sitk包读取nii文件
# tem_png_path = 'D:\\StudyProgram\\Program\\VScode\\Scribble\\XCAD\\test\\scribblepng2nii\\camourflage_00003.png'
# # tem_png_path = 'D:\\StudyProgram\\Program\\VScode\\Scribble\\XCAD\\train_scribble\\Scribble\\15664_22.png'
# nii1 = sitk.ReadImage(tem_png_path)
# nii2 = sitk.GetArrayFromImage(nii1)
# arr = cv2.imread(tem_png_path)#cv2.IMREAD_ANYDEPTH参数用于告诉 cv2.imread() 函数在读取图像时考虑所有可能的位深度，包括 uint16。这样就可以正确地读取 uint16 类型的图像。
# print(nii2)
# nii = sitk.GetImageFromArray(nii2)
# sitk.WriteImage(nii,'camourflage_00003.nii')
# #尝试将nii文件转为正常可显示的png
# image1 = nii2
# image1 = np.where(image1>0.5,1,0)
# print(image1)
# image1 = Image.fromarray(image1.astype(dtype=np.uint8))
# image1.save('image1.png')
# image2 = (nii2*255).astype(dtype=np.uint8)
# print(image2)
# image2 = Image.fromarray(image2)
# image2.save('image2.png')
# plt.imsave('999.jpg',nii2)

