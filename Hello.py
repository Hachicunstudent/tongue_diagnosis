from official.vision.ops.preprocess_ops import resize_and_crop_image
import cv2 
import tensorflow as tf

category_index={
    1: {
        'id': 1,
        'name': 'Tongue'
       }
}


HEIGHT, WIDTH = 640, 640
IMG_SIZE = [HEIGHT, WIDTH, 3]


def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image


import streamlit as st
import os
import tempfile

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(uploaded_file.read())
            return f.name
    except Exception as e:
       
        return None

st.title("Tongue Diagnosis Model")
st.header("Make by Dr. Vu Duc Dai in VietNam")
st.subheader("Application for LucKhi")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path is not None:
        st.info("Image analysis")
        progress = st.progress(0, text="Image processing, please wait")
        img_path=file_path
        #Load model
        model_dect_dir="tongue_obj_dect_model"
        imported = tf.saved_model.load(model_dect_dir)
        model_fn = imported.signatures['serving_default']
        # Dự đoán từ model
        from PIL import Image
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle


        

        # Load and preprocess the image
        image_raw = Image.open(img_path)
        image_np = np.array(image_raw)

        input_image_size = (HEIGHT, WIDTH)  # Specify desired height and width
        image1 = build_inputs_for_object_detection(image_np, input_image_size)
        image = tf.expand_dims(image1, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        progress.progress(1/5) 

        # Predict using your model
        result = model_fn(image)

        boxes=result['detection_boxes'][0,0].numpy()
        score=result['detection_scores'][0,0].numpy()
        

        #Show image        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        image_np = image1.numpy()
        image_np = (image_np).astype('uint8')
        #plt.imshow(image_np)
        
       


        y_min, x_min, y_max, x_max = boxes

        fig, ax = plt.subplots(1)

        image_np_crop = image_np[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Chuyển đổi từ BGR sang RGB
        image_np_crop = cv2.cvtColor(image_np_crop, cv2.COLOR_BGR2RGB)

        # Tạo một tệp tạm thời
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        crop_img_path = tfile.name + '.png'
        # Lưu hình ảnh cropped vào tệp tạm thời
        cv2.imwrite(crop_img_path, image_np_crop)
        # Hiển thị hình ảnh sử dụng Streamlit
        progress.progress(2/5)  # Cập nhật progress bar sau bước 2

        #Load segment model
        
        model_seg_dir="seg_model"
        imported = tf.saved_model.load(model_seg_dir)
        model_sg = imported.signatures['serving_default']
        
        
        

        
        
        def load_image(image_path, height, width):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [height, width])
            img = tf.cast(img, dtype=tf.uint8)
            return img

        def create_binary_mask(pred_mask):
            pred_mask = tf.math.argmax(pred_mask, axis=-1)
            pred_mask = tf.expand_dims(pred_mask, axis=-1)  # thêm một kênh màu
            return tf.cast(pred_mask, dtype=tf.uint8)  # chuyển đổi về dạng uint8

        def segment_object(image, mask):
            mask_rgb = tf.repeat(mask, 3, axis=-1)
            return image * mask_rgb
        
        HEIGHT, WIDTH=(128,128)        
        image = load_image(crop_img_path, HEIGHT, WIDTH)

        # Dự đoán
        predicted_mask = model_sg(tf.expand_dims(image, axis=0))
        binary_mask = create_binary_mask(predicted_mask['logits'])

        # Tạo ảnh đã được segment
        segmented_img = segment_object(image, binary_mask)
        segmented_img = tf.squeeze(segmented_img)        
        # Lưu và hiển thị hình ảnh đã được segment
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            seg_img_path = f.name
            tf.keras.preprocessing.image.save_img(seg_img_path, segmented_img)
        
        progress.progress(3/5)  # Cập nhật progress bar sau bước 3
        
        #Show the result
        import cv2
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt

        img = cv2.imread(crop_img_path)

        # Chuyển đổi ảnh từ BGR sang RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Chuyển đổi ảnh từ RGB sang HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Điều chỉnh giá trị Hue
        (h, s, v) = cv2.split(img_hsv)
        h_adjusted = np.where(h < 20, h + 160, h - 20)
        img_hsv_adjusted = cv2.merge((h_adjusted, s, v))

        # Chuyển đổi lại từ HSV sang RGB
        img_rgb_adjusted = cv2.cvtColor(img_hsv_adjusted, cv2.COLOR_HSV2RGB)
        
        
        # Tính toán phần tử object
        # @title Tính toán phần tử Object detection
        import cv2
        import numpy as np

        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (r, g, b) = cv2.split(img_rgb)

        # Convert the image from RGB to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Extract the Hue, Saturation, and Value channels
        (h, s, v) = cv2.split(img_hsv)

        # Apply CLAHE to V channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_v = clahe.apply(v)

        # Merge the H, S, and enhanced V back
        img_enhanced_hsv = cv2.merge([h, s, clahe_v])

        # Convert the enhanced HSV image back to RGB
        img_enhanced_rgb = cv2.cvtColor(img_enhanced_hsv, cv2.COLOR_HSV2RGB)

        # Increase sharpness
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp_img = cv2.filter2D(img_enhanced_rgb, -1, kernel_sharpen)
        # 4) Show the masks for red, green, and blue colors
        # Create a mask for each color and apply them
        mask_red = cv2.inRange(img_rgb, (100, 0, 0), (255, 100, 100))
        mask_green = cv2.inRange(img_rgb, (0, 100, 0), (100, 255, 100))
        mask_blue = cv2.inRange(img_rgb, (0, 0, 100), (100, 100, 255))

        # Apply the masks
        red_part = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_red)
        green_part = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_green)
        blue_part = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_blue)

        #plt.tight_layout()
        #plt.show()

        # @title Tính toán segment tạo histogram
        # Convert the image from BGR to RGB
        
        segmented_image=cv2.imread(seg_img_path)
        segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        (r2, g2, b2) = cv2.split(segmented_image_rgb)

        # Convert the image from RGB to HSV
        segmented_image_hsv = cv2.cvtColor(segmented_image_rgb, cv2.COLOR_RGB2HSV)


        # Extract the Hue, Saturation, and Value channels
       
        (h2, s2, v2) = cv2.split(segmented_image_hsv)
        
        progress.progress(4/5)  # Cập nhật progress bar sau bước 3
        # @title In ra hình ảnh bỏ bin đầu
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.backends.backend_pdf import PdfPages

        fig = plt.figure(figsize=(20, 20))

        # Tạo lưới với 4 hàng và 4 cột
        gs = gridspec.GridSpec(4, 4, height_ratios=[2, 1, 1, 1])


        # Ở hàng đầu tiên, tạo một subplot chiếm 2 cột đầu và một chiếm 2 cột cuối
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])

        # Các subplot tiếp theo có thể được thêm vào lưới bình thường
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[1, 3])

        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])
        ax10 = fig.add_subplot(gs[2, 3])


        ax11 = fig.add_subplot(gs[3, 0])
        ax12 = fig.add_subplot(gs[3, 1])
        ax13 = fig.add_subplot(gs[3, 2])
        ax14 = fig.add_subplot(gs[3, 3])



        ax1.imshow(img_rgb)
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image")
        ax1.axis('off')


        ax2.imshow(img_hsv)
        ax2.set_title("HSV Image (viewed in RGB)")
        ax2.axis('off')


        ax3.imshow(img_hsv_adjusted)
        ax3.set_title("HSV adjust Image (viewed in RGB)")
        ax3.axis('off')
        #Hue gragh

        # Tính toán histogram
        n, bins = np.histogram(h2.flatten(), bins=180)

        # Loại bỏ bins đầu tiên
        n = n[1:]
        bins = bins[1:]

        # Vẽ biểu đồ
        ax4.bar(bins[:-1], n, width=np.diff(bins), color="Red", alpha=0.5, align="edge")
        ax4.set_title('Hue')
        ax4.set_xlim([1, 180])
        ax4.set_xticklabels([])
        ax3.axis('off')

        # Tạo gradient image cho kênh Hue
        gradient_img = np.zeros((10, 180, 3), dtype=np.uint8)
        gradient_img[:, :, 0] = np.arange(180)
        gradient_img[:, :, 1] = 255  # Set Saturation to max
        gradient_img[:, :, 2] = 255  # Set Value to max
        gradient_img = cv2.cvtColor(gradient_img, cv2.COLOR_HSV2RGB)
        # Add a color bar under the Hue histogram
        axs_colorbar = plt.axes([ax4.get_position().x0, ax4.get_position().y0-0.016, ax4.get_position().width, 0.015])
        axs_colorbar.imshow(gradient_img, aspect='auto', extent=[0, 180, 0, 1])
        #axs_colorbar.set_xlim([0, 256])
        axs_colorbar.set_yticks([])
        #axs_colorbar.axis('off')

        #Saturation gragh
        n, bins = np.histogram(s2.flatten(), bins=256)

        n = n[1:]
        bins = bins[1:]

        ax5.bar(bins[:-1], n, width=np.diff(bins), color="Blue", alpha=0.5, align="edge")
        ax5.set_title('Saturation')
        ax5.set_xlim([1, 256])

        #Value gragh
        # Tính toán histogram
        n, bins = np.histogram(v2.flatten(), bins=256)

        n = n[1:]
        bins = bins[1:]

        ax6.bar(bins[:-1], n, width=np.diff(bins), color="Red", alpha=0.5, align="edge")
        ax6.set_title('Value')
        ax6.set_xlim([1, 256])


        ax7.imshow(img_rgb)
        ax7.axis('off')

        #Biểu đồ đỏ
        n, bins = np.histogram(r2.flatten(), bins=256)

        n = n[1:]
        bins = bins[1:]

        ax8.bar(bins[:-1], n, width=np.diff(bins), color="Red", alpha=0.5, align="edge")
        ax8.set_title( 'Red Histogram')
        ax8.set_xlim([1, 256])


        #Biểu đồ xanh Green
        n, bins = np.histogram(g2.flatten(), bins=256)

        n = n[1:]
        bins = bins[1:]

        ax9.bar(bins[:-1], n, width=np.diff(bins), color="Green", alpha=0.5, align="edge")
        ax9.set_title( 'Green Histogram')
        ax9.set_xlim([1, 256])

        #Biểu đồ xanh Blue
        # Tính toán histogram
        n, bins = np.histogram(b2.flatten(), bins=256)

        # Loại bỏ bins đầu tiên
        n = n[1:]
        bins = bins[1:]

        # Vẽ biểu đồ
        ax10.bar(bins[:-1], n, width=np.diff(bins), color="Blue", alpha=0.5, align="edge")
        ax10.set_title( 'Blue Histogram')
        ax10.set_xlim([1, 256])


        ax11.imshow(img_rgb)
        ax11.axis('off')


        ax12.imshow(img_enhanced_rgb)
        ax12.set_title("Max Contrast")
        ax12.axis('off')

        ax13.imshow(sharp_img)
        ax13.set_title("Max Sharpness")
        ax13.axis('off')

        ax14.imshow(red_part)
        ax14.set_title("Red Mask")
        ax14.axis('off')
        
        st.pyplot(plt)
        progress.progress(5/5) 
        from io import BytesIO
        # Lưu biểu đồ vào một BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        # Nút tải biểu đồ về dưới dạng ảnh
        st.download_button(
            label="Download infographic",
            data=buffer,
            file_name='tongue_infographic.png',
            mime='image/png'
                            )
        



        
        
        
        
                

        
        

        
        


        
        
        
        
        

