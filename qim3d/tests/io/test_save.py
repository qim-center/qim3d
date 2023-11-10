import qim3d
import tempfile
import numpy as np
import os
import hashlib
import pytest
import re

def test_image_exist():
    # Create random test image
    test_image = np.random.randint(0,256,(100,100,100),'uint8')

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,"test_image.tif")

        # Save to temporary directory
        qim3d.io.save(image_path,test_image)

        # Assert that test image has been saved
        assert os.path.exists(image_path)

def test_compression():
    # Get test image (should not be random in order for compression to function)
    test_image = qim3d.examples.blobs_256x256

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,"test_image.tif")
        compressed_image_path = os.path.join(temp_dir,"compressed_test_image.tif")

        # Save to temporary directory with and without compression
        qim3d.io.save(image_path,test_image)
        qim3d.io.save(compressed_image_path,test_image,compression=True)

        # Compute file sizes
        file_size = os.path.getsize(image_path)
        compressed_file_size = os.path.getsize(compressed_image_path)

        # Assert that compressed file size is smaller than non-compressed file size
        assert compressed_file_size < file_size

def test_image_matching():
    # Create random test image
    original_image = np.random.randint(0,256,(100,100,100),'uint8')

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,"original_image.tif")

        # Save to temporary directory
        qim3d.io.save(image_path,original_image)

        # Load from temporary directory
        saved_image = qim3d.io.load(image_path)

        # Get hashes
        original_hash = calculate_image_hash(original_image)
        saved_hash = calculate_image_hash(saved_image)

        # Assert that original image is identical to saved_image
        assert original_hash == saved_hash

def test_compressed_image_matching():
    # Get test image (should not be random in order for compression to function)
    original_image = qim3d.examples.blobs_256x256

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,"original_image.tif")

        # Save to temporary directory
        qim3d.io.save(image_path,original_image,compression=True)

        # Load from temporary directory
        saved_image_compressed = qim3d.io.load(image_path)

        # Get hashes
        original_hash = calculate_image_hash(original_image)
        compressed_hash = calculate_image_hash(saved_image_compressed)

        # Assert that original image is identical to saved_image
        assert original_hash == compressed_hash

def test_file_replace():
    # Create random test image
    test_image1 = np.random.randint(0,256,(100,100,100),'uint8')
    test_image2 = np.random.randint(0,256,(100,100,100),'uint8')

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,"test_image.tif")

        # Save first test image to temporary directory
        qim3d.io.save(image_path,test_image1)
        # Get hash
        hash1 = calculate_image_hash(qim3d.io.load(image_path))
        
        # Replace existing file
        qim3d.io.save(image_path,test_image2,replace=True)
        # Get hash again
        hash2 = calculate_image_hash(qim3d.io.load(image_path))

        # Assert that the file was modified by checking if the second modification time is newer than the first
        assert hash1 != hash2

def test_file_already_exists():
    # Create random test image
    test_image1 = np.random.randint(0,256,(100,100,100),'uint8')
    test_image2 = np.random.randint(0,256,(100,100,100),'uint8')

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,"test_image.tif")

        # Save first test image to temporary directory
        qim3d.io.save(image_path,test_image1)

        with pytest.raises(ValueError,match="A file with the provided path already exists. To replace it set 'replace=True'"):
            # Try to save another image to the existing path
            qim3d.io.save(image_path,test_image2)

def test_no_file_ext():
    # Create random test image
    test_image = np.random.randint(0,256,(100,100,100),'uint8')

    # Create filename without extension
    filename = 'test_image'

    message = 'Please provide a file extension if you want to save as a single file. Otherwise, please provide a basename to save as a TIFF stack'

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,filename)

        with pytest.raises(ValueError,match=message):
            # Try to save the test image to a path witout file extension
            qim3d.io.save(image_path,test_image)

def test_folder_doesnt_exist():
    # Create random test image
    test_image = np.random.randint(0,256,(100,100,100),'uint8')

    # Create invalid path 
    invalid_path = os.path.join('this','path','doesnt','exist.tif')

    #message = f'The directory {re.escape(os.path.dirname(invalid_path))} does not exist. Please provide a valid directory'
    message = f"""The directory {re.escape(os.path.dirname(invalid_path))} does not exist. Please provide a valid directory or specify a basename
 if you want to save a tiff stack as several files to a folder that does not yet exist"""

    with pytest.raises(ValueError,match=message):
        # Try to save test image to an invalid path
        qim3d.io.save(invalid_path,test_image)
    
def test_unsupported_file_format():
    # Create random test image
    test_image = np.random.randint(0,256,(100,100,100),'uint8')

    # Create filename with unsupported format
    filename = 'test_image.unsupported'
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir,filename)    

        with pytest.raises(ValueError,match='Unsupported file format'):
            # Try to save test image with an unsupported file extension
            qim3d.io.save(image_path,test_image)

def test_no_basename():
    # Create random test image
    test_image = np.random.randint(0,256,(100,100,100),'uint8')

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        message = f"""Please provide a basename with the keyword argument 'basename' if you want to save
 a TIFF stack as several files to '{re.escape(temp_dir)}'. Otherwise, please provide a path with a valid filename"""

        with pytest.raises(ValueError,match=message):
            # Try to save test image to an existing directory (indicating
            # that you want to save as several files) without providing a basename
            qim3d.io.save(temp_dir,test_image)

def test_mkdir_tiff_stack():
    # Create random test image
    test_image = np.random.randint(0,256,(10,100,100),'uint8')

    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define a folder that does not yet exist
        path2save= os.path.join(temp_dir,'tempfolder')
       
        # Save to this folder with a basename
        qim3d.io.save(path2save,test_image,basename='test')

        # Assert that folder is created
        assert os.path.isdir(path2save)

def test_tiff_stack_naming():
    # Create random test image
    test_image = np.random.randint(0,256,(10,100,100),'uint8')

    # Define expected filenames
    basename = 'test'
    expected_filenames = [basename + str(i).zfill(2) + '.tif' for i,_ in enumerate(test_image)]
    
    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        qim3d.io.save(temp_dir,test_image,basename=basename)

        assert expected_filenames==sorted(os.listdir(temp_dir))
    

def test_tiff_stack_slicing_dim():
    # Create random test image where the three dimensions are not the same length
    test_image = np.random.randint(0,256,(5,10,15),'uint8')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Iterate thorugh all three dims and save the image as slices in 
        # each dimension in separate folder and assert the number of files
        # match the shape of the image
        for dim in range(3):
            path2save = os.path.join(temp_dir,'dim'+str(dim))
            qim3d.io.save(path2save,test_image,basename='test',sliced_dim=dim)
            assert len(os.listdir(path2save))==test_image.shape[dim]

def calculate_image_hash(image): 
    image_bytes = image.tobytes()
    hash_object = hashlib.md5(image_bytes)
    return hash_object.hexdigest()