# MNIST PNGs

I made this so I can have simpler example code.

```cpp
import cv2
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
for i in range(1000):
    cv2.imwrite(f'path/to/train-{i:04}-{Y_train[i]}.png', X_train[i])
```
