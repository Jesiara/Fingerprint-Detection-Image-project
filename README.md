# Fingerprint-Detection-Image-project

<html>
<head>
    <style>
        p { text-align: justify; }
    </style>
</head>
<body>
    <p>
        This project matches fingerprints using Gaussian filtering, adaptive thresholding, and morphological operations. 
        Minutiae points are extracted and compared via Euclidean distance, considering image rotations for accuracy.
    </p>
</body>
</html>

1. The project involves key steps: image acquisition, preprocessing, feature extraction, and matching.
2. Preprocessing includes image rotation, Gaussian filtering, adaptive thresholding, and morphological operations.
3. Morphological operations like closing and skeletonization enhance fingerprint structures.
4. The Zhang-Suen thinning algorithm is used for skeletonization to obtain a one-pixel-wide fingerprint skeleton.
5. Minutiae points, such as ridge endings and bifurcations, are extracted from the skeletonized image.
6. Euclidean distance is used to match minutiae points between two fingerprint images.
7. The matching percentage is calculated based on the number of matched minutiae points.
8. The system accounts for image rotations to improve accuracy and robustness.
9. The approach demonstrates the importance of preprocessing and feature extraction in fingerprint recognition for security applications.
