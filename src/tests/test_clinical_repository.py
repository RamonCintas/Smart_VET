import json
import re
import shutil
from pathlib import Path

import pytest

from clinical_repository import ClinicalRepository, PatientNotFoundError


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "src" / "back-end" / "fastapi" / "data"
BENCHMARK_PATH = ROOT_DIR / "src" / "tests" / "fixtures" / "benchmark_qa_gemma.json"


def test_repository_loads_validated_clinical_data():
    repository = ClinicalRepository(DATA_DIR)
    repository.load()

    metadata = repository.metadata()
    patients = repository.list_patients()
    context = repository.get_context("vet-001")

    assert metadata["patient_count"] == 50
    assert metadata["medication_count"] == 37
    assert len(metadata["data_version"]) == 12
    assert len(patients) == 50
    assert context["metadata"]["patient_id"] == "VET-001"
    assert "MED-001" in context["metadata"]["matched_medication_ids"]


def test_repository_rejects_unknown_patient():
    repository = ClinicalRepository(DATA_DIR)

    with pytest.raises(PatientNotFoundError):
        repository.get_context("VET-999")


def test_repository_reloads_changed_patient_file(tmp_path):
    temporary_data = tmp_path / "data"
    temporary_data.mkdir()
    shutil.copy(DATA_DIR / "pacientes_clinicos.json", temporary_data)
    shutil.copy(DATA_DIR / "medicamentos.xlsx", temporary_data)

    repository = ClinicalRepository(temporary_data, reload_on_change=True)
    original = repository.get_context("VET-001")

    patients_path = temporary_data / "pacientes_clinicos.json"
    patients = json.loads(patients_path.read_text(encoding="utf-8"))
    patients[0]["reavaliacao"] = "Registro atualizado para teste"
    patients_path.write_text(
        json.dumps(patients, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    updated = repository.get_context("VET-001")

    assert updated["patient"]["reavaliacao"] == "Registro atualizado para teste"
    assert updated["metadata"]["data_version"] != original["metadata"]["data_version"]


def test_benchmark_references_every_known_patient():
    repository = ClinicalRepository(DATA_DIR)
    known_ids = {patient["id"] for patient in repository.list_patients()}
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    referenced_ids = {
        re.search(r"VET-\d{3}", item["caso_referencia"]).group(0)
        for item in benchmark
    }

    assert len(benchmark) == 50
    assert referenced_ids == known_ids
