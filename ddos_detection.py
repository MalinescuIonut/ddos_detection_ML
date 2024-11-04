import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhoogla/cicddos2019")

print("Path to dataset files:", path)