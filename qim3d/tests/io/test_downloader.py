import qim3d
import os

def test_download():
    folder = 'Cowry_Shell'
    file = 'Cowry_DOWNSAMPLED.tif'
    path = os.path.join(folder,file)

    dl = qim3d.io.Downloader()

    dl.Cowry_Shell.Cowry_DOWNSAMPLED()


    img = qim3d.io.load(path)

    # Remove temp file
    os.remove(path)
    os.rmdir(folder)

    assert img.shape == (500,350,350)

def test_get_file_size_right():
    coal_file = 'https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository/Coal/CoalBrikett.tif'
    size = qim3d.io.downloader._get_file_size(coal_file)

    assert size == 2_400_082_900


def test_get_file_size_wrong():
    file_to_folder = 'https://archive.compute.dtu.dk/files/public/projects/viscomp_data_repository/'
    size = qim3d.io.downloader._get_file_size(file_to_folder)

    assert size == -1


def test_extract_html():
    url = 'https://archive.compute.dtu.dk/files/public/projects/viscomp_data_repository'
    html = qim3d.io.downloader._extract_html(url)
    
    assert 'data-path="/files/public/projects/viscomp_data_repository"' in html
