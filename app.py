import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mediapipe as mp
import rembg
import io

# --- Configuration ---
PASSPORT_WIDTH = 600  # pixels (2 inches at 300dpi)
PASSPORT_HEIGHT = 600 # pixels (2 inches at 300dpi)
HEAD_HEIGHT_RATIO_MIN = 0.5  # Head height should be 50% to 69% of image height
HEAD_HEIGHT_RATIO_MAX = 0.69
EYE_OPEN_RATIO_THRESHOLD = 0.2 # Threshold to determine if eyes are open

# --- Helper Classes ---

class PassportValidator:
    def __init__(self):
        self.face_mesh = None
        self.initialization_error = None

        # Initialize MediaPipe Face Mesh when available.
        # Some builds expose only the Tasks API and do not include mp.solutions.
        mp_solutions = getattr(mp, "solutions", None)
        mp_face_mesh = getattr(mp_solutions, "face_mesh", None) if mp_solutions else None

        if mp_face_mesh is None:
            self.initialization_error = (
                "Face validation is unavailable: this MediaPipe build does not include "
                "Face Mesh Solutions (mp.solutions). Install a MediaPipe build that "
                "includes Face Mesh support for full validation."
            )
            return

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )

    def analyze_face(self, image_np):
        """
        Analyzes the face for passport compliance.
        Returns: (success_boolean, list_of_errors, landmark_dict)
        """
        errors = []

        if self.face_mesh is None:
            return False, [self.initialization_error], None

        h, w, _ = image_np.shape
        results = self.face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return False, ["No face detected. Please upload a clearer photo."], None

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Key Landmarks (Indices approximate standard MediaPipe Face Mesh)
        # Nose Tip: 1
        # Left Eye: 33 (inner), 133 (outer), 159 (top), 145 (bottom)
        # Right Eye: 362 (inner), 263 (outer), 386 (top), 374 (bottom)
        # Chin: 152
        # Top of head approximation: 10 (hairline area)
        
        nose_tip = landmarks[1]
        left_eye_inner = landmarks[33]
        right_eye_inner = landmarks[362]
        
        # Head Size & Position Checks
        face_top = landmarks[10] # Approx hairline
        chin = landmarks[152]
        
        head_height_pixels = abs(face_top.y - chin.y) * h
        image_height_pixels = h
        
        # 1. Check Head Size (Vertical)
        ratio = head_height_pixels / image_height_pixels
        if ratio < HEAD_HEIGHT_RATIO_MIN:
            errors.append(f"Head is too small (should be ~50-70% of height). Currently: {int(ratio*100)}%")
        elif ratio > HEAD_HEIGHT_RATIO_MAX:
            errors.append(f"Head is too large (should be ~50-70% of height). Currently: {int(ratio*100)}%")

        # 2. Check Head Rotation (Yaw - looking left/right)
        # Compare distance from nose tip to left eye inner vs right eye inner
        dist_left = abs(nose_tip.x - left_eye_inner.x)
        dist_right = abs(nose_tip.x - right_eye_inner.x)
        
        if abs(dist_left - dist_right) > 0.05: # Threshold for rotation
             errors.append("Head is rotated. Please face the camera directly.")

        # 3. Check Eyes Open (Simple check using vertical eye distance)
        # Left Eye Vertical
        l_eye_top = landmarks[159]
        l_eye_bot = landmarks[145]
        l_eye_dist = abs(l_eye_top.y - l_eye_bot.y)
        
        # Right Eye Vertical
        r_eye_top = landmarks[386]
        r_eye_bot = landmarks[374]
        r_eye_dist = abs(r_eye_top.y - r_eye_bot.y)

        # If eye opening is extremely small relative to face width
        face_width = abs(landmarks[33].x - landmarks[263].x)
        if (l_eye_dist / face_width) < 0.02 or (r_eye_dist / face_width) < 0.02:
            errors.append("Eyes appear to be closed or mostly closed.")

        return len(errors) == 0, errors, landmarks

class ImageProcessor:
    @staticmethod
    def remove_background(image_pil):
        """Removes background and replaces with white."""
        output_image = rembg.remove(image_pil)
        # Create a white background
        background = Image.new("RGB", output_image.size, (255, 255, 255))
        # Paste the cutout onto the white background using the alpha mask
        background.paste(output_image, mask=output_image.convert("RGBA").split()[3])
        return background

    @staticmethod
    def crop_and_center(image_pil, landmarks):
        """
        Crops the image to passport specs based on eye position.
        US Passport: 
        - Top of head should have some space.
        - Eyes should be located 1-1/8" to 1-3/8" from the bottom.
        """
        img_np = np.array(image_pil)
        h, w = img_np.shape[:2]
        
        # Landmarks are normalized [0.0 - 1.0]
        left_eye_x = landmarks[33].x * w
        right_eye_x = landmarks[263].x * w
        left_eye_y = landmarks[33].y * h
        right_eye_y = landmarks[263].y * h
        
        # Calculate center of eyes
        eye_center_x = (left_eye_x + right_eye_x) / 2
        eye_center_y = (left_eye_y + right_eye_y) / 2
        eye_distance = abs(left_eye_x - right_eye_x)

        # Calculate zoom factor based on eye distance.
        # For a 600x600 image, eye distance is usually around 100-120px.
        # We want the head to take up about 60% of vertical space.
        # Let's scale so the eye distance dictates the crop size.
        # Rough heuristic: Face height is approx 2.5x eye distance.
        estimated_face_h = eye_distance * 2.5
        
        # Calculate the crop box.
        # The US Dept of State specifies eyes should be between 1-1/8" (337px) and 1-3/8" (412px) 
        # from the bottom of a 2x2 (600x600px) image.
        # Let's aim for the middle: 375px from bottom.
        
        target_img_size = 600
        target_eye_y_from_bottom = 375 
        
        # Determine scale factor
        # We need to map the current eye distance to the desired visual proportions.
        # Let's guess a scale factor that maps the estimated face height to ~65% of the target image height.
        scale = (target_img_size * 0.65) / estimated_face_h
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        scaled_img = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        eye_center_x_scaled = eye_center_x * scale
        eye_center_y_scaled = eye_center_y * scale
        
        # Calculate where to crop to get a 600x600 image
        # We want the eyes to be at pixel (Y = 600 - 375 = 225) from top
        desired_eye_y = target_img_size - target_eye_y_from_bottom
        
        # Calculate top-left corner of the crop
        crop_x = int(eye_center_x_scaled - target_img_size / 2)
        crop_y = int(eye_center_y_scaled - desired_eye_y)
        
        # Create a white canvas of target size
        final_img = Image.new("RGB", (target_img_size, target_img_size), (255, 255, 255))
        
        # Paste the scaled image onto the canvas.
        # If the crop goes out of bounds, we paste what we have.
        paste_x = -crop_x
        paste_y = -crop_y
        
        # We need to handle the case where the source image is smaller than the crop or shifted oddly
        # For simplicity, we paste the relevant part of the scaled image onto the center if calculations go wild
        box = (max(0, -paste_x), max(0, -paste_y), min(new_w, target_img_size - paste_x), min(new_h, target_img_size - paste_y))
        
        final_img.paste(scaled_img.crop(box), (max(0, paste_x), max(0, paste_y)))
        
        return final_img

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Passport Photo Validator", layout="wide")
    
    st.title("📷 US Passport Photo Validator & Processor")
    st.write("Upload a selfie to automatically remove the background, check for compliance, and generate a passport-ready photo.")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Selfie")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load Image
        image_pil = Image.open(uploaded_file).convert("RGB")
        
        # Keep as RGB for MediaPipe analysis
        image_np = np.array(image_pil)

        with st.spinner("Analyzing face..."):
            # 1. Validate Face
            validator = PassportValidator()
            is_valid, errors, landmarks = validator.analyze_face(image_np)

        with st.spinner("Removing background... (this may take a few seconds)"):
            # 2. Process Image (Remove BG)
            processed_pil = ImageProcessor.remove_background(image_pil)

            # 3. Crop & Center
            if landmarks:
                # Recalculate landmarks on the processed image? 
                # For simplicity, we use the original landmark coordinates to determine crop ratio
                # as background removal doesn't change geometry.
                processed_pil = ImageProcessor.crop_and_center(processed_pil, landmarks)
            else:
                # Fallback: Just resize if no face detected
                processed_pil = processed_pil.resize((PASSPORT_WIDTH, PASSPORT_HEIGHT), Image.Resampling.LANCZOS)

        # Display Results
        with col1:
            st.image(image_pil, caption="Original Upload", width="stretch")

        with col2:
            st.header("Result")
            
            # Display Validation Status
            if is_valid:
                st.success("✅ Photo meets basic technical requirements!")
            else:
                st.error("⚠️ Validation Issues Detected:")
                for err in errors:
                    st.warning(f"- {err}")
            
            st.image(processed_pil, caption="Processed Passport Photo (2x2 @ 300dpi)", width=300)

            # Download Button
            buf = io.BytesIO()
            processed_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Passport Photo",
                data=byte_im,
                file_name="passport_photo.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()