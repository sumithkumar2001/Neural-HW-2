Sumith kumar kotagiri (700759479)

### README for Convolutional Neural Network (CNN) Code

#### Overview

This project demonstrates the implementation of several key tasks in image processing and deep learning. The tasks are divided into four main sections:

1. **Convolution Operation**
2. **Edge Detection Using Sobel Filter**
3. **Pooling Operations (Max and Average Pooling)**
4. **Building Deep Learning Models (AlexNet & ResNet)**

Each task highlights different concepts, including convolution, edge detection, pooling, and constructing CNN architectures.

---

### Task 1: Convolution Operation

In this task, we implement a **2D Convolution** operation on a given input matrix (image). The process involves applying a 3x3 **kernel/filter** to the input matrix to generate an **output feature map**.

Key parameters:
- **Stride**: Determines the step size for sliding the kernel over the input matrix.
- **Padding**: Specifies how the input matrix is padded before applying the kernel. Two options are available: `SAME` and `VALID`.
  - **SAME** padding: Adds zero-padding around the input matrix so the output feature map has the same spatial dimensions as the input.
  - **VALID** padding: No padding is added; the output feature map dimensions are smaller than the input matrix.

The function `convolve2d` performs the convolution operation on the input matrix using the specified kernel, stride, and padding options.

```python
convolve2d(input_matrix, kernel, stride, padding)
```

---

### Task 2: Edge Detection Using Sobel Filter

This task applies the **Sobel filter** for edge detection. The Sobel operator computes the gradient of the image intensity at each pixel, highlighting regions of high spatial frequency which typically correspond to edges in the image.

- **Sobel-X**: Detects edges in the horizontal direction.
- **Sobel-Y**: Detects edges in the vertical direction.

The `apply_sobel_filter` function loads an image from the given path, applies the Sobel filter in both the x and y directions, and then displays the original image along with the edge-detected results.

```python
apply_sobel_filter("path_to_image")
```

---

### Task 3: Max Pooling and Average Pooling

Pooling is a downsampling operation commonly used in CNNs to reduce the spatial dimensions of feature maps and extract dominant features. This task demonstrates two types of pooling:

1. **Max Pooling**: In each pool, the maximum value is selected.
2. **Average Pooling**: In each pool, the average value is selected.

The functions `max_pooling` and `average_pooling` implement these operations for a given input matrix with a specified pool size.

```python
max_pooling(input_matrix, pool_size)
average_pooling(input_matrix, pool_size)
```

---

### Task 4: Deep Learning Models

This section demonstrates the construction of two popular **Convolutional Neural Network (CNN)** architectures: **AlexNet** and **ResNet**.

#### AlexNet

AlexNet is a deep CNN architecture with several convolutional and fully connected layers. It is widely used for image classification tasks. The model consists of:

- Convolutional layers with ReLU activations.
- Max-pooling layers to reduce spatial dimensions.
- Fully connected layers followed by a softmax activation for classification.

```python
create_alexnet()
```

#### ResNet

ResNet is a deep CNN architecture that includes **residual blocks**, which allow the network to train deeper models by skipping certain layers (using shortcut connections). This architecture helps to avoid the vanishing gradient problem.

- **Residual Block**: Consists of two convolutional layers, followed by an addition of the input tensor (skip connection).
- The model ends with a softmax output for classification.

```python
create_resnet()
```

Both models can be compiled and trained using Keras or TensorFlow libraries for image classification tasks.

---

### Requirements

To run this project, you need the following libraries:
- **NumPy**: For matrix manipulation.
- **OpenCV**: For image processing and applying the Sobel filter.
- **TensorFlow & Keras**: For building the CNN models.

Install the required libraries using `pip`:
```bash
pip install numpy opencv-python tensorflow
```

---

### Example Usage

1. **Convolution**: You can change the kernel, stride, and padding to experiment with different configurations.
2. **Sobel Filter**: Call `apply_sobel_filter("path_to_image")` with the path to an image to see edge detection results.
3. **Pooling**: Use `max_pooling` or `average_pooling` on any matrix to perform pooling.
4. **AlexNet/ResNet**: Create and summarize the models by calling `create_alexnet()` or `create_resnet()`.


### Acknowledgements

- **OpenCV** for image processing.
- **TensorFlow** and **Keras** for building CNN models.
- **NumPy** for matrix manipulation in convolution and pooling operations.

