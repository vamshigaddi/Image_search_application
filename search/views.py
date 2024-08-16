from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from .utils import ImageSimilarity
import os

# Set the path to the folder containing images
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the path to the FAISS index and metadata files within the app directory
INDEX_FILE = os.path.join(APP_DIR, 'faiss_index')
METADATA_FILE = os.path.join(APP_DIR, 'image_paths.pkl')

# Initialize the ImageSimilarity class with the full paths
similarity_model = ImageSimilarity(index_file=INDEX_FILE, metadata_file=METADATA_FILE)

def index_view(request):
    return render(request, 'index.html')

def search_similar_images(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['image']

            # Save the uploaded image to a temporary file
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
            with open(temp_path, 'wb+') as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)

            # Load the similarity index and fetch similar images
            similarity_model.load_index()
            results = similarity_model.fetch_similar(temp_path, top_k=6)
            print("Results: ", results)

            # Remove the temporary file
            #os.remove(temp_path)

            # Convert results to the format expected by the template
            formatted_results = []
            for result in results:
                # Check if the score is greater than 0.55
                if result[1] > 0.10:
                    # Convert absolute path to a relative path
                    rel_path = os.path.relpath(result[0], APP_DIR)
                    
                    # Adjust the URL to match MEDIA_URL and relative path
                    url_path = os.path.join(settings.MEDIA_URL, 'images', os.path.basename(rel_path)).replace('\\', '/')
                    print("Formatted URL: ", url_path)  # Debug statement
                    
                    formatted_results.append({'url': url_path, 'score': result[1]})

            # Create URL for the uploaded image
            uploaded_image_url = os.path.join(settings.MEDIA_URL, 'temp_image.jpg').replace('\\', '/')

            return render(request, 'results.html', {
                'results': formatted_results,
                'uploaded_image': {'url': uploaded_image_url}
            })
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
