import streamlit as st
import os

from PIL import Image
from predict_skin import load_model, predict_skin

def main():
    # load model
    load_model()

    st.title(" Skin Segmentation and Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        #  load image, save, show
        image = Image.open(uploaded_file)
        path_save_image = os.getcwd() + '/Results/image_input.jpg'
        image.save(path_save_image)
        st.image(image, caption="Input Image", use_column_width=True)

        # predict
        Predicted = predict_skin(path_save_image)

        if Predicted == 'Check your image again!':
            st.markdown(f"<h2>Something Wrong: {Predicted}</h2>", unsafe_allow_html=True)
            return
        # show result with bigger font size
        st.markdown(f"<h2>Predicted result: {Predicted}</h2>", unsafe_allow_html=True)

        # show Results/image_integrate_bbox.jpg
        image_bbox = Image.open(os.getcwd() + '/Results/image_integrate_bbox.jpg')
        st.markdown(f"<h3>Segmentation Output</h3>", unsafe_allow_html=True)
        st.image(image_bbox, caption="BBox (None Padding and Have Padding) of Image and Mask", use_column_width=True)
        
        # show Results/image_compare.jpg
        st.markdown(f"<h3>Classification Input Compare</h3>", unsafe_allow_html=True)
        image_compare = Image.open(os.getcwd() + '/Results/Image_compare.jpg')
        st.image(image_compare, caption="Input Original vs Input After Cropt", use_column_width=True)

if __name__ == "__main__":
    main()
