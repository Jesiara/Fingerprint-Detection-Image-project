# Fingerprint-Detection-Image-project

<p style="text-align: justify;">
    The project focuses on fingerprint matching using image processing techniques like Gaussian filtering, adaptive thresholding, and morphological operations. Minutiae points (ridge endings and bifurcations) are extracted and matched using Euclidean distance to determine fingerprint similarity. The system improves accuracy by considering image rotations and calculates a match percentage for verification.
</p>

1. The project involves key steps: image acquisition, preprocessing, feature extraction, and matching.
2. Preprocessing includes image rotation, Gaussian filtering, adaptive thresholding, and morphological operations.
3. Morphological operations like closing and skeletonization enhance fingerprint structures.
4. The Zhang-Suen thinning algorithm is used for skeletonization to obtain a one-pixel-wide fingerprint skeleton.
5. Minutiae points, such as ridge endings and bifurcations, are extracted from the skeletonized image.
6. Euclidean distance is used to match minutiae points between two fingerprint images.
7. The matching percentage is calculated based on the number of matched minutiae points.
8. The system accounts for image rotations to improve accuracy and robustness.
9. The approach demonstrates the importance of preprocessing and feature extraction in fingerprint recognition for security applications.

[Download PDF](https://github.com/Jesiara/Fingerprint-Detection-Image-project/blob/main/image_project_report.pdf)
