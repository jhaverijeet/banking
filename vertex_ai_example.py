import vertexai
from vertexai.generative_models import GenerativeModel

# TODO: Replace with your actual Google Cloud Project ID and Region
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "us-central1" # e.g. us-central1, europe-west2

def generate_text(project_id: str, location: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    
    # Load the model
    model = GenerativeModel("gemini-1.5-pro")
    
    # Prompt the model
    prompt = "Explain why the Net Present Value (NPV) is important in banking in 2 short sentences."
    print(f"Sending prompt: '{prompt}'")
    
    # Get the response
    response = model.generate_content(prompt)
    
    return response.text

if __name__ == "__main__":
    if PROJECT_ID == "YOUR_PROJECT_ID":
        print("Please update the PROJECT_ID variable with your Google Cloud Project ID.")
    else:
        try:
            print("Connecting to Vertex AI...")
            result = generate_text(PROJECT_ID, REGION)
            print("\nResponse from Vertex AI:")
            print(result)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("\nMake sure you have:")
            print("1. Installed the SDK: pip install google-cloud-aiplatform")
            print("2. Authenticated: gcloud auth application-default login")
