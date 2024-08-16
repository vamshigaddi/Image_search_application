# Image Search App

## Project Description

The Image Search App is a Django-based web application that allows users to upload images and perform searches based on the uploaded images. This app utilizes image classification and search functionalities to find and display relevant images.

## Features

- **Image Upload**: Users can upload images to the application.
- **Image Search**: Perform searches based on the uploaded images.
- **Image Classification**: Classify uploaded images using a pre-trained model.
- **User Interface**: Simple and intuitive interface to interact with the application.

## Technologies Used

- **Django**: Web framework used for the backend.
- **Python**: Programming language.
- **Bootstrap**: For styling the frontend.
  
 # DEMO:
   https://www.loom.com/share/4da9035a0da344419587e248c919ce57?sid=dca8acf1-42cf-40b9-9bbe-953d5c2fd2a0

## Installation

### Prerequisites

- Python 3.11 
- pip (Python package installer)
- virtualenv (recommended for virtual environment management)

### Steps to Install

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vamshigaddi/Image_search_application.git
2. **Create and Activate a Virtual Environment**
   ```bash
    python -m venv myvenv
    source myvenv/bin/activate  
    #On Windows use
    myvenv\Scripts\activate
3. **Requirements**
   ```bash
   pip install -r requirements.txt
4. **Apply Migrations**
   ```bash
   python manage.py migrate
5. **Run the development server**
   ```bash
   python manage.py runserver
   ```
####  Access the Application
- Open your web browser and go to http://127.0.0.1:8000/ to access the application
6. **Usage**
- Upload an Image: Navigate to the upload page and select an image to upload.
- Search for Images: After uploading an image, you can perform a search to find similar images.
- View Results: Search results will be displayed based on the classification and search functionality.


