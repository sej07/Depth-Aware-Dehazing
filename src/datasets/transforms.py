from torchvision import transforms


def get_train_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # Resize slightly larger
        transforms.RandomCrop(image_size),  # Random crop for augmentation
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.ToTensor(),  # Convert to tensor [0, 1]
    ])


def get_test_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class PairedTransform:
    
    def __init__(self, image_size=256, is_train=True):
        self.image_size = image_size
        self.is_train = is_train
        
    def __call__(self, hazy_img, clean_img):
        # Resize
        if self.is_train:
            resize_size = self.image_size + 32
        else:
            resize_size = self.image_size
            
        resize = transforms.Resize((resize_size, resize_size))
        hazy_img = resize(hazy_img)
        clean_img = resize(clean_img)
        
        # Random crop
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                hazy_img, output_size=(self.image_size, self.image_size)
            )
            hazy_img = transforms.functional.crop(hazy_img, i, j, h, w)
            clean_img = transforms.functional.crop(clean_img, i, j, h, w)
        
        # Random horizontal flip
        if self.is_train and transforms.RandomHorizontalFlip(p=0.5):
            import random
            if random.random() > 0.5:
                hazy_img = transforms.functional.hflip(hazy_img)
                clean_img = transforms.functional.hflip(clean_img)
        
        # To tensor
        to_tensor = transforms.ToTensor()
        hazy_img = to_tensor(hazy_img)
        clean_img = to_tensor(clean_img)
        
        return hazy_img, clean_img