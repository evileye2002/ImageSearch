import sys, os, math, pickle, numpy as np, matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image


# Ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(
        inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output
    )
    return extract_model


# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Trich dac trung
    vector = model.predict(img_tensor)[0]
    # Chuan hoa vector = chia chia L2 norm (tu google search)
    vector = vector / np.linalg.norm(vector)
    return vector


def store_vectors(
    dataset_folder="data/dataset",
    vectors_file="data/vectors.pkl",
    paths_file="data/paths.pkl",
):
    model = get_extract_model()
    vectors, paths = []

    for image_path in os.listdir(dataset_folder):
        image_path_full = os.path.join(dataset_folder, image_path)
        image_vector = extract_vector(model, image_path_full)
        vectors.append(image_vector)
        paths.append(image_path_full)

    pickle.dump(vectors, open(vectors_file, "wb"))
    pickle.dump(paths, open(paths_file, "wb"))


def search_image(
    img_path,
    vectors_file="data/vectors.pkl",
    paths_file="data/paths.pkl",
    k=4,
):
    if not img_path:
        return

    model = get_extract_model()
    search_vector = extract_vector(model, img_path)
    vectors = pickle.load(open(vectors_file, "rb"))
    paths = pickle.load(open(paths_file, "rb"))

    distance = np.linalg.norm(vectors - search_vector, axis=1)
    ids = np.argsort(distance)[:k]
    nearest_image = [(paths[id], distance[id]) for id in ids]

    axes = []
    grid_size = int(math.sqrt(k))
    fig = plt.figure(figsize=(10, 5))

    for id in range(k):
        draw_image = nearest_image[id]
        axes.append(fig.add_subplot(grid_size, grid_size, id + 1))
        axes[-1].set_title(nearest_image[id])
        plt.imshow(Image.open(draw_image[0]))

    fig.tight_layout()
    plt.show()


def test(test_img="data/test/test_1.png"):
    store_vectors()
    search_image(test_img)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_img = sys.argv[2] if len(sys.argv) > 2 else "data/test/test_1.png"

        test(test_img)
    else:
        print("Usage: python main.py test [test_img]")
