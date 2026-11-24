from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageFilesDataset(Dataset):
    """
    Creates torch Dataset from directory of images.
    Must be structured as dir/<class>/<img_name>.<extension> for `conditional=True`
    For `conditional=False`, will search recursively for all files that match the extension
    """

    def __init__(self, path, name=None, extension="png", transform=None, conditional=False):
        self.path = path
        self.name = name
        self.extension = extension

        self.conditional = conditional  # If conditional, will get the class from the parent folder's name
        self.transform = transform
        self.files = []

        self.files_loaded = False  # For lazy loading of files

    def load_files(self):
        for curr_path in Path(self.path).rglob(f"*.{self.extension}"):
            if self.conditional:
                self.files.append((curr_path, curr_path.parent.name))
            else:
                self.files.append((curr_path, 0))
        self.files_loaded = True

    def __len__(self):
        if not self.files_loaded:
            self.load_files()

        return len(self.files)

    # Retain the upstream idx keyword rather than Dataset's index parameter.
    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        if not self.files_loaded:
            self.load_files()

        img_path, class_id = self.files[idx]
        with Image.open(img_path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)
            return img, class_id

    def get_class(self, idx):
        return self.files[idx][1]
