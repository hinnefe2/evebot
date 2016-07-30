class WindowNotFoundException(Exception):
    """Exception for when a specified window isn't found"""

class TemplateNotFoundException(Exception):
    """Exception for when the specified template can't be found on the screen"""
    pass

class NoOCRMatchException(Exception):
    """Exception for when order information extracted via OCR can't be matched to orders"""
    pass
