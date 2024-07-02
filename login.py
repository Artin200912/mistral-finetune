from huggingface_hub import login

# Replace 'your_hugging_face_token' with your actual token
token = "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"
login(token, add_to_git_credential=True)

print("Successfully logged in to Hugging Face Hub and saved the token to git credential helper!")
