# tests/test_profiling.py

import sys
from pathlib import Path
# Proje kök dizinini path'e ekle
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from data_handling import generate_data_profile

def test_generate_data_profile_file(tmp_path):
    # Küçük örnek DataFrame
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    # Geçici klasörde çıktı dosyası yolu
    output_file = tmp_path / "feedback_profile.html"
    
    # Profil raporu oluştur
    generate_data_profile(df, output_path=str(output_file))
    
    # Dosyanın oluşturulduğunu doğrula
    assert output_file.exists(), "Profil raporu dosyası oluşturulmamış."
    assert output_file.stat().st_size > 0, "Profil raporu dosyası boş."
