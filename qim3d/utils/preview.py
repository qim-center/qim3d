import numpy as np
from PIL import Image

# These are fixed because of unicode characters bitmaps. 
# It could only be flexible if each character had a function that generated the bitmap based on size
X_STRIDE = 4
Y_STRIDE = 8 


BACK_TO_NORMAL = "\u001b[0m"
END_MARKER = -10

"""
For each unicode character that we can print (and is not inverse of another unicode character)
there is a numnber which serves as a bitmap. That bitmap says how does the unicode character looks
like in a field 4x8.
"""
BITMAPS = [
    # Block graphics
    # 0xffff0000, 0x2580,  // upper 1/2; redundant with inverse lower 1/2
    0x00000000, '\u00a0',
    0x0000000f, '\u2581',  # lower 1/8
    0x000000ff, '\u2582',  # lower 1/4
    0x00000fff, '\u2583', 
    0x0000ffff, '\u2584',  # lower 1/2
    0x000fffff, '\u2585', 
    0x00ffffff, '\u2586',  # lower 3/4
    0x0fffffff, '\u2587',
    # 0xffffffff, 0x2588,  # full; redundant with inverse space

    0xeeeeeeee, '\u258a',  # left 3/4
    0xcccccccc, '\u258c',  # left 1/2
    0x88888888, '\u258e',  # left 1/4

    0x0000cccc, '\u2596',  # quadrant lower left
    0x00003333, '\u2597',  # quadrant lower right
    0xcccc0000, '\u2598',  # quadrant upper left
    # 0xccccffff, 0x2599,  # 3/4 redundant with inverse 1/4
    0xcccc3333, '\u259a',  # diagonal 1/2
    # 0xffffcccc, 0x259b,  # 3/4 redundant
    # 0xffff3333, 0x259c,  # 3/4 redundant
    0x33330000, '\u259d',  # quadrant upper right
    # 0x3333cccc, 0x259e,  # 3/4 redundant
    # 0x3333ffff, 0x259f,  # 3/4 redundant

    # Line drawing subset: no double lines, no complex light lines

    0x000ff000, '\u2501',  # Heavy horizontal
    0x66666666, '\u2503',  # Heavy vertical

    0x00077666, '\u250f',  # Heavy down and right
    0x000ee666, '\u2513',  # Heavy down and left
    0x66677000, '\u2517',  # Heavy up and right
    0x666ee000, '\u251b',  # Heavy up and left

    0x66677666, '\u2523',  # Heavy vertical and right
    0x666ee666, '\u252b',  # Heavy vertical and left
    0x000ff666, '\u2533',  # Heavy down and horizontal
    0x666ff000, '\u253b',  # Heavy up and horizontal
    0x666ff666, '\u254b',  # Heavy cross

    0x000cc000, '\u2578',  # Bold horizontal left
    0x00066000, '\u2579',  # Bold horizontal up
    0x00033000, '\u257a',  # Bold horizontal right
    0x00066000, '\u257b',  # Bold horizontal down

    0x06600660, '\u254f',  # Heavy double dash vertical

    0x000f0000, '\u2500',  # Light horizontal
    0x0000f000, '\u2500',  #
    0x44444444, '\u2502',  # Light vertical
    0x22222222, '\u2502',

    0x000e0000, '\u2574',  # light left
    0x0000e000, '\u2574',  # light left
    0x44440000, '\u2575',  # light up
    0x22220000, '\u2575',  # light up
    0x00030000, '\u2576',  # light right
    0x00003000, '\u2576',  # light right
    0x00004444, '\u2577',  # light down
    0x00002222, '\u2577',  # light down

    0x11224488, '\u2571',  # diagonals
    0x88442211, '\u2572',
    0x99666699, '\u2573',

    0, END_MARKER, 0  # End marker 
]

class Color:
    def __init__(self, red:int, green:int, blue:int):
        self.check_value(red)
        self.check_value(green)
        self.check_value(blue)
        self.red = red
        self.green = green
        self.blue = blue

    def check_value(sel, value:int):
        assert isinstance(value, int), F"Color value has to be integer, this is {type(value)}"
        assert value < 256, F"Color value has to be between 0 and 255, this is {value}"
        assert value >= 0, F"Color value has to be between 0 and 255, this is {value}"
    
    def __str__(self):
        """
        Returns the string in ansi color format
        """
        return F"{self.red};{self.green};{self.blue}"


def chardata(unicodeChar: str, character_color:Color, background_color:Color) -> str:
    """
    Given the character and colors, it creates the string, which when printed in terminal simulates pixels.
    """
    # ESC[38;2;⟨r⟩;⟨g⟩;⟨b⟩ m Select RGB foreground color
    # ESC[48;2;⟨r⟩;⟨g⟩;⟨b⟩ m Select RGB background color
    assert isinstance(character_color, Color)
    assert isinstance(background_color, Color)
    assert isinstance(unicodeChar, str)
    return F"\033[38;2;{character_color}m\033[48;2;{background_color}m{unicodeChar}"

def get_best_unicode_pattern(bitmap:int) -> tuple[int, str, bool]:
    """
    Goes through the list of unicode characters and looks for the best match for bitmap representing the given segment
    It computes the difference by counting 1s after XORing the two. If they are identical, the count will be 0.
    This character will be printed

    Parameters:
    -----------
    - bitmap (int): int representing the bitmap the image segment.

    Returns:
    ----------
    - best_pattern (int): int representing the pattern that was the best match, is then used to calculate colors
    - unicode (str): the unicode character that represents the given bitmap the best and is then printed
    - inverse (bool): The list does't contain unicode characters that are inverse of each other. The match can be achieved by simply using 
        the inversed bitmap. But then we need to know if we have to switch background and foreground color.
    """
    best_diff = 8
    best_pattern = 0x0000ffff
    unicode = '\u2584'
    inverse = False

    bit_not = lambda n: (1 << 32) - 1 - n 

    i = 0
    while BITMAPS[i+1] != END_MARKER:
        pattern = BITMAPS[i]
        for j in range(2):
            diff = (pattern ^ bitmap).bit_count()
            if diff < best_diff:
                best_pattern = pattern
                unicode = BITMAPS[i+1]
                best_diff = diff
                inverse = bool(j)
            pattern = bit_not(pattern)

        i += 2

    return best_pattern, unicode, inverse
    
def int_bitmap_from_ndarray(array_bitmap:np.ndarray)->int:
    """
    Flattens the array
    Changes all numbers to strings
    Creates a string representing binary number
    Casts it to integer
    """
    return int(F"0b{''.join([str(i) for i in array_bitmap.flatten()])}", base = 2)

def ndarray_from_int_bitmap(bitmap:int, shape:tuple = (8, 4))-> np.ndarray:
    """
    Gets the binary representation
    Gets rid of leading '0b
    Fill in leading zeros so its correct length
    Make it list of integers
    Make it numpy array
    """
    string = str(bin(bitmap))[2:].zfill(shape[0] * shape[1])
    return np.array([int(i) for i in string]).reshape(shape)
    
def create_bitmap(image_segment:np.ndarray)->int:
    """
    Parameters:
    ------------
    image_segment: np.ndarray of shape (x, y, 3)

    Returns:
    ----------
    bitmap: int, each bit says if the unicode character should cover this bit or not
    """

    max_color = np.max(np.max(image_segment, axis=0), axis = 0)
    min_color = np.min(np.min(image_segment, axis=0), axis = 0)
    rng = np.absolute(max_color - min_color)
    max_index = np.argmax(rng)
    if np.sum(rng) == 0:
        return 0
    split_threshold = rng[max_index]/2 + min_color[max_index]
    bitmap = np.array(image_segment[:, :, max_index] <= split_threshold, dtype = int)


    return int_bitmap_from_ndarray(bitmap)

def get_color(image_segment:np.ndarray, char_array:np.ndarray) -> Color:
    """
    Computes the average color of the segment from pixels specified in charr_array
    The color is then average over the part then unicode character covers or the background

    Parameters:
    -----------
    - image_segment: 4x8 part of the image with the original values so average color can be calculated
    - char_array: indices saying which pixels out of the 4x8 should be used for color calculation

    Returns:
    ---------
    - color: containing the average color over defined pixels
    """
    colors = []
    for channel_index in range(image_segment.shape[2]):
        channel = image_segment[:,:,channel_index]
        colors.append(int(np.average(channel[char_array])))

    return Color(colors[0], colors[1], colors[2]) if len(colors) == 3 else Color(colors[0], colors[0], colors[0])

def get_colors(image_segment:np.ndarray, char_array:np.ndarray) -> tuple[Color, Color]:
    """
    Parameters:
    ----------
    - image_segment
    - char_array


    Returns:
    ----------
    - Foreground color
    - Background color
    """
    return get_color(image_segment, char_array == 1), get_color(image_segment, char_array == 0)

def segment_string(image_segment:np.ndarray)-> str:
    """
    Creates bitmap so its best represent the color distribution
    Finds the best match in unicode characters
    If the best match is character taking up the whole field, then both colors are the same (it doesn't matter)
    If the best match was inverted unicode character, background and foreground colors need to be switched,
        otherwise it is not smooth
    Creates and returns the ansi string to be printed
    """
    bitmap = create_bitmap(image_segment)
    bitmap, unicode, reverse = get_best_unicode_pattern(bitmap)
    if unicode == '\u00a0':
        bg_color = fg_color = get_color(image_segment, ndarray_from_int_bitmap(bitmap))
    else:
        fg_color, bg_color = get_colors(image_segment, ndarray_from_int_bitmap(bitmap))
        if reverse:
            bg_color, fg_color = fg_color, bg_color
    return chardata(unicode, fg_color, bg_color)

def image_ansi_string(image:np.ndarray) -> str:
    """
    For each segment 4x8 finds the string with colored unicode character
    Create the string for whole image

    Parameters:
    -----------
    - image: image to be displayed in terminal

    Returns:
    ----------
    - ansi_string: when printed, will render the image
    """
    string = []
    for y in range(0, image.shape[0], Y_STRIDE):
        for x in range(0, image.shape[1], X_STRIDE):

            this_segment = image[y:y+Y_STRIDE, x:x+X_STRIDE, :]
            if this_segment.shape[0] != Y_STRIDE:
                segment = np.zeros((Y_STRIDE, X_STRIDE, this_segment.shape[2]))
                segment[:this_segment.shape[0], :, :] = this_segment
                this_segment = segment
            string.append(segment_string(this_segment))

        string.append(F"{BACK_TO_NORMAL}\n")

    return ''.join(string)




###################################################################
#               Image preparation
###################################################################

def rescale_image(image:np.ndarray, size:tuple)->np.ndarray:
    """
    The unicode bitmaps are hardcoded for 4x8 segments, they cannot be scaled
    Thus the image must be scaled to fit the desired resolution
    """
    if image.shape[2] == 1:
        image = np.squeeze(image)
    image = Image.fromarray(image)
    image = np.array(image.resize(size))
    if image.ndim != 3:
        image = np.expand_dims(image, 2)
    return image


def check_and_adjust_image_dims(image:np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    elif image.ndim == 3:
        if image.shape[2] == 1:  # grayscale image
            pass
        elif image.shape[2] == 3: # colorful image
            pass
        elif image.shape[2] == 4: # contains alpha channel
            image = image[:,:,:3]
        elif image.shape[0] == 3: # torch images have color channels as the first axis
            image = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(F"Image must have 2 (grayscale) or 3 (colorful) dimensions. Yours has {image.ndim}")
    
    return image

def check_and_adjust_values(image:np.ndarray, relative_intensity:bool = True) -> np.ndarray:
    """
    Checks if the values are between 0 and 255
    If not, normalizes the values so they are in that interval

    Parameters:
    -------------
    - image
    - relative_intensity: If maximum values are pretty low, they will be barely visible. If true, it normalizes 
        the values, so that the maximum is at 255

    Returns:
    -----------
    - adjusted_image
    """

    m = np.max(image)
    if m > 255:
        image = np.array(255*image/m, dtype = np.uint8)
    elif m < 1:
        image = np.array(255*image, dtype = np.uint8)

    if relative_intensity:
        m = np.max(image)
        image = np.array((image/m)*255, dtype = np.uint8)

    return image

def choose_slice(image:np.ndarray, axis:int = None, slice:int = None):
    """
    Preview give the possibility to choose axis to be sliced and slice to be displayed
    """
    if axis is not None:
        image = np.moveaxis(image, axis, -1)

    if slice is None:
        slice = image.shape[2]//2
    else:
        if slice > image.shape[2]:
            slice = image.shape[2]-1
    return image[:,:, slice]

###################################################################
#               Main function
###################################################################

def image_preview(image:np.ndarray, image_width:int = 80, axis:int = None, slice:int = None, relative_intensity:bool = True):
    if image.ndim == 3 and image.shape[2] > 4:
        image = choose_slice(image, axis, slice)
    image = check_and_adjust_image_dims(image)
    ratio = X_STRIDE*image_width/image.shape[1]
    image = check_and_adjust_values(image, relative_intensity)
    image = rescale_image(image, (X_STRIDE*image_width, int(ratio * image.shape[0])))
    print(image_ansi_string(image))

