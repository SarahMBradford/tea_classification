from PIL import Image
import streamlit as st
st.title("Meet the Curly Headed Coder")
# create two columns
col1, col2 = st.columns([4,8])
# display image in first column
col1.image(Image.open('headshot.jpg'), use_column_width=True)
col1.image(Image.open('skydiving.jpg'), use_column_width=True)
col1.image(Image.open('hot_air_balloon.jpg'), use_column_width=True)
col2.header("Sarah Bradford")
col2.subheader("Data Scientist")
col2.write("Sarah is a Spelman College alumna and currently pursuing a Masters of Data Science at Michigan State University. She is deeply passionate about the intersection of health and data science, she offers a strong academic foundation in both, which makes her ready and willing to change the world. Sometimes, the mission to change the world involves starting her day with a hot cup of tea. Using her background in research, programming, and various data manipulation techniques, she has extracted features from datasets to make predictions and decisions. Her current aspiration is to contribute her analytical mindset and dedication to improving outcomes within a dynamic data science role, where she can steer innovation and make a positive impact.")
col2.write("She is more than just a data scientist, she is a skydiver, a traveler, a foodie, and a lover of all things spontaneous!")
col2.subheader("Get to Know Her More!")
col2.write("LinkedIn: https://www.linkedin.com/in/sarah-bradford-398806186/")
