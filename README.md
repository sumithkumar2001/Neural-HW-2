Sumith kumar kotagiri (700759479)


## **1. Convolution Operations**

### **Code Overview**
The code performs 2D convolution on a 5x5 input matrix using a 3x3 kernel. The convolution is implemented manually with support for different strides and padding options (`'valid'` and `'same'`).

### **Key Components**
1. **Input Matrix**: A 5x5 matrix with values from 1 to 25.
2. **Kernel**: A 3x3 matrix used for convolution.
3. **Convolution Function**:
   - `convolve2d(input_matrix, kernel, stride, padding)`:
     - Computes the convolution of the input matrix with the kernel.
     - Supports `'valid'` (no padding) and `'same'` (padding to maintain input size) padding modes.
     - Output size is calculated based on the input size, kernel size, stride, and padding.
4. **Results**:
   - The function is called with different combinations of strides (1 and 2) and padding modes (`'valid'` and `'same'`).
   - The output feature maps are printed for each combination.

### **Explanation**
- **Stride**: Determines the step size for sliding the kernel over the input matrix.
- **Padding**:
  - `'valid'`: No padding is applied, resulting in a smaller output size.
  - `'same'`: Padding is applied to ensure the output size matches the input size.
- The convolution operation involves element-wise multiplication of the kernel with overlapping regions of the input matrix, followed by summation.

---

## **2. Edge Detection Using Sobel Filter**

### **Code Overview**
The code applies the Sobel filter to detect edges in a grayscale image. The Sobel filter computes gradients in the horizontal (x) and vertical (y) directions.

### **Key Components**
1. **Sobel Filter**:
   - `cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)`: Computes the gradient in the x-direction.
   - `cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)`: Computes the gradient in the y-direction.
2. **Image Loading**:
   - The image is loaded in grayscale mode using `cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)`.
3. **Display**:
   - The original image and the Sobel-filtered images (x and y directions) are displayed using `cv2.imshow()`.

### **Explanation**
- The Sobel filter is used to highlight edges in an image by detecting intensity changes.
- The x-direction filter detects vertical edges, while the y-direction filter detects horizontal edges.

---

## **3. Pooling Operations**

### **Code Overview**
The code demonstrates two types of pooling operations: **max pooling** and **average pooling**. These operations are applied to a randomly generated 4x4 matrix.

### **Key Components**
1. **Max Pooling**:
   - `max_pooling(input_matrix, pool_size)`:
     - Divides the input matrix into non-overlapping regions of size `pool_size x pool_size`.
     - Computes the maximum value in each region.
2. **Average Pooling**:
   - `average_pooling(input_matrix, pool_size)`:
     - Divides the input matrix into non-overlapping regions of size `pool_size x pool_size`.
     - Computes the average value in each region.
3. **Results**:
   - The original matrix, max-pooled matrix, and average-pooled matrix are printed.

### **Explanation**
- **Pooling** is used to downsample the input matrix, reducing its spatial dimensions while retaining important features.
- **Max Pooling**: Retains the most prominent features in each region.
- **Average Pooling**: Smooths the input matrix by averaging values in each region.

---

## **4. Deep Learning Models (AlexNet and ResNet)**

### **Code Overview**
The code defines two popular deep learning architectures: **AlexNet** and **ResNet**, using TensorFlow and Keras.

### **Key Components**

#### **Task 1: AlexNet**
1. **Architecture**:
   - Convolutional layers with varying filter sizes and strides.
   - Max pooling layers for downsampling.
   - Fully connected (dense) layers with dropout for regularization.
   - Output layer with softmax activation for classification.
2. **Model Summary**:
   - The model summary is printed using `alexnet_model.summary()`.

#### **Task 2: ResNet**
1. **Residual Block**:
   - A custom residual block is defined using `residual_block(input_tensor, filters)`.
   - The block consists of two convolutional layers and a skip connection (addition of the input tensor).
2. **Architecture**:
   - Initial convolutional layer followed by residual blocks.
   - Flattening and fully connected layers for classification.
3. **Model Summary**:
   - The model summary is printed using `resnet_model.summary()`.

### **Explanation**
- **AlexNet**:
  - A classic CNN architecture designed for image classification.
  - Uses large convolutional filters and max pooling to extract features.
- **ResNet**:
  - Introduces residual connections to address the vanishing gradient problem in deep networks.
  - Each residual block learns residual functions, making it easier to train very deep networks.

---

## **How to Run the Code**
1. **Dependencies**:
   - Install the required libraries:
     ```
     pip install numpy opencv-python tensorflow
     ```
2. **Running the Code**:
   - Save the code in a Python file (e.g., `main.py`).
   - Run the file using:
     ```
     python main.py
     ```
3. **Edge Detection**:
   - Replace `"sample_image.jpg"` in `apply_sobel_filter("sample_image.jpg")` with the path to your image file.

---

## **Outputs**
1. **Convolution**:
   - Output feature maps for different stride and padding combinations.
2. **Edge Detection**:
   - Displays the original image and Sobel-filtered images.
3. **Pooling**:
   - Prints the original matrix, max-pooled matrix, and average-pooled matrix.
4. **Deep Learning Models**:
   - Prints the architecture summaries for AlexNet and ResNet.

---

This code provides a comprehensive overview of fundamental concepts in image processing and deep learning, including convolution, edge detection, pooling, and building neural networks.
