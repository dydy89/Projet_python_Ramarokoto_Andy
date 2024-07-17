import pytest
import sys
sys.path.append('C:\\Users\\33768\\Documents\\andy\\Projet_python_Ramarokoto_Andy\\Table_detector')
from detector import TableDetector

@pytest.fixture
def sample_image():
    return r'C:\Users\33768\Pictures\une_facture.jpg'

def test_extract_table_success(sample_image):
    detector = TableDetector()
    tables = detector.extract_table(sample_image, threshold=0.3)  # Utiliser un seuil réduit
    print(f"Tables detected: {tables}")
    assert tables is not None
    assert len(tables) > 0

def test_extract_table_failure():
    detector = TableDetector()
    with pytest.raises(Exception):
        detector.extract_table(r'C:\Users\33768\Pictures\image-inexistante.png')
