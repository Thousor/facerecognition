import cv2
import numpy as np
import dlib
from PIL import Image
import mediapipe as mp

class BeautyProcessor:
    def __init__(self):
        # 初始化mediapipe人脸检测
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 初始化dlib人脸关键点检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 初始化OpenCV人脸检测器
        self.face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')

    def smooth_skin(self, image, level=0.7):
        """
        磨皮美颜
        :param image: 输入图像
        :param level: 磨皮程度 (0-1)
        :return: 处理后的图像
        """
        # 双边滤波磨皮
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 混合原图和双边滤波后的图像，level控制磨皮强度
        # 较高的level意味着更多的双边滤波效果，即更平滑
        smoothed_image = cv2.addWeighted(image, 1 - level, bilateral, level, 0)

        # 锐化以保留细节，但强度要低，避免过度锐化
        # 创建一个锐化核
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        sharpened_image = cv2.filter2D(smoothed_image, -1, kernel_sharpening)
        
        # 再次混合，以控制锐化强度，避免过度锐化
        # 锐化强度可以根据level进行调整，或者设置为一个固定的小值
        result = cv2.addWeighted(smoothed_image, 0.8, sharpened_image, 0.2, 0) # 0.2 is a small sharpening factor
        
        return result

    def whiten_skin(self, image, level=0.3):
        """
        美白处理
        :param image: 输入图像
        :param level: 美白程度 (0-1)
        :return: 处理后的图像
        """
        Color_list = [
            1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
            41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
            76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
            106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
            130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
            151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
            171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
            217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
            228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
            238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
            245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
            251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
            254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 256
        ]

        # 应用双边滤波
        img_filtered = cv2.bilateralFilter(image, 9, 75, 75)

        # 应用查找表进行美白
        result = img_filtered.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j][0] = Color_list[result[i, j][0]]
                result[i, j][1] = Color_list[result[i, j][1]]
                result[i, j][2] = Color_list[result[i, j][2]]
        
        # 根据level调整美白强度
        result = cv2.addWeighted(image, 1 - level, result, level, 0)
        return result

    def remove_acne(self, image):
        """
        祛痘处理
        :param image: 输入图像
        :return: 处理后的图像
        """
        # Apply bilateral filter for initial smoothing while preserving edges
        smoothed_image = cv2.bilateralFilter(image, 9, 75, 75)

        # Apply median blur with a slightly larger kernel for blemish removal
        # A larger kernel can remove more prominent blemishes but might over-smooth
        result = cv2.medianBlur(smoothed_image, 7) # Increased kernel size from 5 to 7

        return result

    # 双线性插值法
    def BilinearInsert(self, src, ux, uy):
        h, w, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = x1 + 1
            y1 = int(uy)
            y2 = y1 + 1

            # 确保坐标在图像范围内
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h - 1)

            part1 = src[y1, x1].astype(np.float32) * (float(x2) - ux) * (float(y2) - uy)
            part2 = src[y1, x2].astype(np.float32) * (ux - float(x1)) * (float(y2) - uy)
            part3 = src[y2, x1].astype(np.float32) * (float(x2) - ux) * (uy - float(y1))
            part4 = src[y2, x2].astype(np.float32) * (ux - float(x1)) * (uy - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.uint8)

    # 局部平移算法
    def localTranslationWarp(self, srcImg, startX, startY, endX, endY, radius):
        ddradius = float(radius * radius)
        copyImg = srcImg.copy()

        # 计算公式中的|m-c|^2
        ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
        
        # 优化：只处理形变圆的边界框内的像素
        minX = max(0, int(startX - radius))
        maxX = min(srcImg.shape[1], int(startX + radius))
        minY = max(0, int(startY - radius))
        maxY = min(srcImg.shape[0], int(startY + radius))

        for i in range(minX, maxX):
            for j in range(minY, maxY):
                distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

                if (distance < ddradius):
                    ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                    ratio = ratio * ratio

                    UX = i - ratio * (endX - startX)
                    UY = j - ratio * (endY - startY)

                    value = self.BilinearInsert(srcImg, UX, UY)
                    copyImg[j, i] = value

        return copyImg

    def slim_face(self, image, level=0.2):
        """
        瘦脸处理 (使用局部平移算法)
        :param image: 输入图像
        :param level: 瘦脸程度 (0-1)
        :return: 处理后的图像
        """
        img_copy = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return image

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_node = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Define the target point for the warp (chin point)
            chin_point = landmarks_node[8]

            # Iterate through jawline points (landmarks 0 to 16)
            # and apply the warp to pull them towards the chin point.
            # Exclude the chin point itself (landmark 8) from being a start_point.
            for i in range(0, 17): # Landmarks 0 to 16
                if i == 8: # Skip the chin point itself
                    continue

                start_point = landmarks_node[i]
                
                # Calculate the radius of influence for the warp
                # The radius should be larger for points further from the chin,
                # and scaled by the slimming level.
                # A larger level means a larger radius of influence and stronger pull.
                base_radius = np.linalg.norm(start_point - chin_point) * 0.3 # Base radius as a fraction of distance to chin
                radius = int(base_radius * (1 + level * 1.5)) # Scale radius by level, with a stronger multiplier

                # Apply the local translation warp
                # The warp pulls the area around start_point towards chin_point
                img_copy = self.localTranslationWarp(img_copy, start_point[0], start_point[1], chin_point[0], chin_point[1], radius)

        return img_copy

    def process_image(self, image, options):
        """
        处理图片文件
        :param image: 输入图片 (numpy array)
        :param options: 美颜选项字典
        :return: 处理后的图像
        """
        result = self.process_frame(image, options)
        return result

    def process_frame(self, frame, options):
        """
        处理单帧图像
        :param frame: 输入帧
        :param options: 美颜选项字典
        :return: 处理后的帧
        """
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray) # Use dlib's face detector

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            points = []
            for i in range(0, landmarks.num_parts):
                points.append((landmarks.part(i).x, landmarks.part(i).y))
            
            # Create a convex hull mask of the face
            mask = np.zeros_like(frame, dtype=np.uint8)
            if len(points) > 0:
                # Extend the mask upwards to include the forehead
                # Find the topmost point and extend a bit further up
                topmost_y = min([p[1] for p in points])
                # Estimate forehead height based on face height
                forehead_extension = int(h * 0.3) # Extend by 30% of face height
                
                # Create a new set of points for the extended mask
                extended_points = []
                for p in points:
                    extended_points.append((p[0], p[1]))
                
                # Add points for the top corners of the forehead area
                # Estimate width of forehead based on face width
                forehead_width_factor = 0.8 # Adjust as needed
                forehead_left_x = int(x + w * (1 - forehead_width_factor) / 2)
                forehead_right_x = int(x + w * (1 + forehead_width_factor) / 2)
                
                extended_points.append((forehead_left_x, max(0, topmost_y - forehead_extension)))
                extended_points.append((forehead_right_x, max(0, topmost_y - forehead_extension)))

                hull = cv2.convexHull(np.array(extended_points))
                cv2.fillConvexPoly(mask, hull, (255, 255, 255))
            
            # Smooth the mask edges using Gaussian blur
            mask = cv2.GaussianBlur(mask, (51, 51), 0) # Larger kernel for smoother transition
            
            # Convert mask to float for blending
            mask_float = mask.astype(np.float32) / 255.0

            # Apply effects only within the face region defined by the mask
            processed_face_region = frame.copy()

            if options.get('smooth', False):
                processed_face_region = self.smooth_skin(processed_face_region, options.get('smooth_level', 0.7))
            
            if options.get('whiten', False):
                processed_face_region = self.whiten_skin(processed_face_region, options.get('whiten_level', 0.3))
                
            if options.get('remove_acne', False):
                processed_face_region = self.remove_acne(processed_face_region)
            
            # Blend the processed face region with the original image using the smoothed mask
            result = (frame * (1 - mask_float) + processed_face_region * mask_float).astype(np.uint8)

        # Slim face is applied globally as it's a deformation
        if options.get('slim', False):
            result = self.slim_face(result, options.get('slim_level', 0.2))
            
        return result 