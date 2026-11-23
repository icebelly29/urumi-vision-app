# Camera Calibration FAQ

## Question: How is the camera calibrated, and how does it work with different cameras and lenses?

### Current Calibration Method

This application does not use saved camera parameters. It uses dynamic planar calibration with an ArUco frame.

1. **Physical Markers**
   - Four ArUco markers are installed at known physical coordinates on the cutting bed.
   - The default frame measures $775\text{ mm} \times 1065\text{ mm}$.

2. **Corner Detection**
   - The application finds the center coordinates of all four corner markers in the photo.

3. **Homography Calculation**
   - OpenCV calculates a perspective transform matrix (`cv.getPerspectiveTransform`).
   - The matrix maps the marker coordinates in pixels to their known positions in millimeters.

4. **Perspective Warping**
   - The application flattens the photo to a top-down view (`cv.warpPerspective`).
   - The warped image has a constant scale factor ($\approx 2.58\text{ pixels per millimeter}$).

---

### How Different Cameras and Lenses Function

The homography method works automatically with different cameras. You do not need to configure sensor size or focal length. 

However, optical differences between lenses change measurement accuracy:

#### 1. Standard Lenses (1x Mode)
- **Behavior:** These lenses have low optical distortion.
- **Result:** Straight lines remain straight across the full cutting bed.
- **Recommendation:** Use this lens mode for the best dimensional accuracy.

#### 2. Wide-Angle and Fisheye Lenses (0.5x Mode)
- **Behavior:** These lenses cause radial barrel distortion.
- **Problem:** A four-point homography only corrects perspective tilt. It does not correct curved optical lines.
- **Result:** Corners align accurately, but straight lines bend near the center and edges of the bed.
- **Recommendation:** Do not use ultra-wide lenses.

#### 3. Camera Resolution
- **High Resolution:** Gives more pixels per millimeter. Vector paths and small features become more precise.
- **Low Resolution:** Decreases vector precision. Small text and thin lines can disappear.

---

### Instructions for Best Results

- Use the main camera sensor at 1x magnification.
- Hold the camera directly above the center of the bed.
- Ensure all four ArUco markers are clear and visible.
- Avoid extreme camera angles.
- If you must use a wide-angle lens, remove radial distortion first with OpenCV `cv.undistort` and a calibration grid.
