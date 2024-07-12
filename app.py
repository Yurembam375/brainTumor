
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image # type: ignore

# Load the saved model
model = keras.models.load_model('model/brainTumor_model.keras')

# Define the target size based on your model's input size (150x150)
target_size = (150, 150)

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('L')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to make predictions
def predict(image):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    return predictions[0][0]

def main():
    st.title('Brain Tumor Detection')
    st.markdown(
        """
        <style>
            .title {
                font-family: 'Arial', sans-serif;
                text-align: center;
                color: #2c3e50;
            }
            .file-upload-btn {
                display: block;
                width: 100%;
                max-width: 300px;
                margin: 20px auto;
                padding: 10px;
                background-color: #3498db;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .file-upload-btn:hover {
                background-color: #2980b9;
            }
            .predict-btn {
                display: block;
                width: 100%;
                max-width: 200px;
                margin: 20px auto;
                padding: 10px;
                background-color: #27ae60;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .predict-btn:hover {
                background-color: #219a52;
            }
            .prediction-text {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin-top: 20px;
            }
            .description {
                font-size: 18px;
                font-style: italic;
                text-align: center;
                margin-top: 10px;
                color: white;
            }
            .resource-link {
                font-size: 16px;
                text-align: center;
                margin-top: 20px;
                color: #2980b9;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f1f1f1;
                color: #808080;
                text-align: center;
                padding: 10px 0;
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Upload an MRI image and click on the **Predict** button to determine if it contains a tumor or not."
    )

    uploaded_file = st.file_uploader("Choose an MRI Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', width=300)

        if st.button('Predict', key='predict_button'):
            prediction = predict(image)
            predicted_class = 'Healthy' if prediction > 0.5 else 'Tumor'
            color = "green" if predicted_class == 'Healthy' else "red"
            st.markdown(f"<p class='prediction-text'>Prediction: <span style='color:{color};'>{predicted_class}</span></p>", unsafe_allow_html=True)

            if predicted_class == 'Tumor':
                st.markdown("<p class='description'>Description: The model predicts that the MRI image contains a tumor.</p>", unsafe_allow_html=True)
                # Display types of brain tumors
                st.header("Types of Brain Tumors")
                st.write("""
                There are several types of brain tumors, including:
                * **Gliomas:** Tumors that arise from glial cells, which support nerve cells. Examples include astrocytomas, oligodendrogliomas, and glioblastomas.
                * **Meningiomas:** Tumors that develop from the meninges, the protective membranes covering the brain and spinal cord.
                * **Pituitary adenomas:** Tumors that occur in the pituitary gland.
                * **Schwannomas:** Tumors that arise from Schwann cells, which produce the myelin sheath that covers nerves. An example is acoustic neuroma.
                * **Medulloblastomas:** Common pediatric brain tumors that start in the lower part of the brain and can spread to the spinal cord.
                * **Craniopharyngiomas:** Rare tumors that start near the pituitary gland.
                """)
                # Display links to resources for brain tumor treatment
                st.header("Resources for Brain Tumor Treatment")
                st.markdown("[Mayo Clinic - Brain Tumor Treatment](https://www.mayoclinic.org/tests-procedures/brain-tumor-treatment/pyc-20384880)")
                st.markdown("[Johns Hopkins Medicine - Brain Tumor Center](https://www.hopkinsmedicine.org/neurology_neurosurgery/centers_clinics/brain_tumor/)")
                st.markdown("[Cleveland Clinic - Brain Tumor Treatment](https://my.clevelandclinic.org/health/diseases/6142-brain-tumors)")
                st.markdown("[Dana-Farber Cancer Institute - Brain Tumor Center](https://www.dana-farber.org/brain-tumor-center/)")
            else:
                st.markdown("<p class='description'>Description: The model predicts that the MRI image does not contain a tumor.</p>", unsafe_allow_html=True)
                # Recommendation for maintaining a healthy brain
                st.header("Tips for Maintaining a Healthy Brain")
                st.write("A healthy diet plays a crucial role in maintaining brain health. Here are some foods that are beneficial for brain health:")
                st.markdown("* **Fatty Fish:** Rich in omega-3 fatty acids, essential for brain health. Examples include salmon, trout, and sardines.")
                st.markdown("* **Berries:** Packed with antioxidants, which may help reduce inflammation and oxidative stress.")
                st.markdown("* **Nuts and Seeds:** High in antioxidants, healthy fats, and vitamin E, all of which may benefit brain health.")
                st.markdown("* **Leafy Greens:** Rich in antioxidants and vitamin K, important for brain health.")
                st.markdown("* **Whole Grains:** Provide a steady supply of energy for the brain.")
                st.markdown("* **Avocados:** Rich in healthy fats and vitamin E, which are important for brain health.")
                st.markdown("* **Dark Chocolate:** Contains flavonoids, caffeine, and antioxidants, which may improve brain function.")
                st.markdown("* **Green Tea:** Contains caffeine and antioxidants, which may enhance brain function and improve mood.")

    # Add the footer
    st.markdown('<div class="footer">© 2024 Brain Tumor Detection</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
