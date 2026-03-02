from PIL import Image

class BaseGenerator:
    """
    Base interface for on-the-fly image generators.
    All custom generators should inherit from this class and implement the generate() method.
    """
    def __init__(self, **kwargs):
        """
        Initialize the generator.
        
        Args:
            **kwargs: Arbitrary keyword arguments. Generators should accept **kwargs
                      to gracefully handle arguments intended for other generators.
        """
        pass

    def generate(self) -> Image.Image:
        """
        Generate a single mixing image.
        
        Returns:
            PIL.Image: A generated RGB image (typically size 224x224).
        """
        raise NotImplementedError("Subclasses must implement the generate() method.")