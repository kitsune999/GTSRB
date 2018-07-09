from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.preprocessing import Normalizer

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

img = load_img('Final_Training/Images/00000/00000_00024.ppm')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# print x.shape
# x = x.astype('float32')
# scale = Normalizer().fit(x[:,:,0])
# x[:,:,0] = scale.transform(x[:,:,0])
# scale = Normalizer().fit(x[:,:,1])
# x[:,:,1] = scale.transform(x[:,:,1])
# scale = Normalizer().fit(x[:,:,2])
# x[:,:,2] = scale.transform(x[:,:,2])

x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

datagen.fit(x)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='00000', save_format='ppm'):
    if i < 1:
        print batch
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
